import gc
import traceback
import os
from copy import deepcopy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from llm_attacks import AttackPrompt, MultiPromptAttack, PromptManager
from llm_attacks import get_embedding_matrix, get_embeddings

class UnlikelihoodLoss(nn.Module):
    def __init__(self):
        super(UnlikelihoodLoss, self).__init__()
        self.eps = 0.01

    def forward(self, input, target):
        probabilities = F.softmax(input, dim=1)
        p = probabilities.gather(1, target.unsqueeze(1))
        p = torch.clamp(p, min=0, max=1-self.eps)       # avoid some inf loss case
        loss = -torch.log(1 - p)
        loss = torch.clamp(loss, min=0, max=10)         # avoid some inf loss case
        return loss.squeeze()

def apply_cosine_decay(tensor: torch.Tensor) -> torch.Tensor:
    '''to conduct cosine decay on the last dimension'''
    L = tensor.size(-1)
    decay_weights = 0.5 + 0.5 * torch.cos(torch.linspace(0, 0.5*torch.pi, L, device=tensor.device, dtype=tensor.dtype))
    return tensor * decay_weights

def token_gradients_dsn_loss(model, input_ids, input_slice, target_slice, loss_slice, test_prefixes, test_token_length, test_prefixes_toks, alpha, use_decay=False, use_refusal=False):
    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1,
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)

    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
                        [
                            embeds[:,:input_slice.start,:],
                            input_embeds,
                            embeds[:,input_slice.stop:,:]
                        ],
                        dim=1)

    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]

    affirmative_loss = nn.CrossEntropyLoss(reduction='none')(logits[0,loss_slice,:], targets)
    if use_decay:   # cosine decay could only be utlized towards vanilla target loss
        affirmative_loss = apply_cosine_decay(affirmative_loss)
    affirmative_loss = torch.mean(affirmative_loss)

    if not use_refusal:
        loss = affirmative_loss
    else:
        refusal_loss = 0
        count_loss = 0
        crit = UnlikelihoodLoss()
        for j_in_algorithm in range(len(test_prefixes)):
            key_word_length = test_token_length[j_in_algorithm]

            for loss_start in range(loss_slice.start , 99999):
                if loss_start + key_word_length > logits.shape[1]:
                    break
                refusal_loss_slice = slice(loss_start, loss_start+key_word_length)
                bs = logits.shape[0]
                cross_loss_target = torch.tensor(test_prefixes_toks[j_in_algorithm]).unsqueeze(0).to(logits.device)
                cross_loss_target = cross_loss_target.repeat(bs, 1)

                temp_loss = crit(logits[:,refusal_loss_slice,:].transpose(1,2), cross_loss_target)

                refusal_loss += temp_loss.mean()    # the refusal loss won't utilize cosine decay
                count_loss += 1

        refusal_loss = refusal_loss/count_loss
        loss = affirmative_loss + alpha * refusal_loss

    loss.backward()
    return one_hot.grad.clone()

class DSNAttackPrompt(AttackPrompt):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def grad(self, model):
        if not self.para.use_different_aug_sampling_alpha:
            sampling_alpha = self.para.augmented_loss_alpha
        else:
            sampling_alpha = self.para.aug_sampling_alpha2
        use_refusal = True if self.para.use_aug_sampling else False

        return token_gradients_dsn_loss(
            model,
            self.input_ids.to(model.device),
            self._control_slice,
            self._target_slice,
            self._loss_slice,
            self.test_prefixes,
            self.test_token_length,
            self.test_prefixes_toks,
            sampling_alpha,
            use_decay=self.para.use_target_loss_cosine_decay,
            use_refusal=use_refusal
        )

class DSNPromptManager(PromptManager):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def sample_control(self, grad, batch_size, topk=256, temp=1, allow_non_ascii=True):

        if not allow_non_ascii:
            grad[:, self._nonascii_toks.to(grad.device)] = np.infty
        top_indices = (-grad).topk(topk, dim=1).indices
        control_toks = self.control_toks.to(grad.device)
        original_control_toks = control_toks.repeat(batch_size, 1)

        new_token_pos = torch.arange(
            0,
            len(control_toks),
            len(control_toks) / batch_size,
            device=grad.device
        ).type(torch.int64)

        new_token_val = torch.gather(
            top_indices[new_token_pos], 1,
            torch.randint(0, topk, (batch_size, 1),
            device=grad.device)
        )
        new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

        return new_control_toks

class DSNMultiPromptAttack(MultiPromptAttack):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def step(self,
            batch_size=1024,
            topk=256,
            temp=1,
            allow_non_ascii=True,
            target_weight=1,
            control_weight=0.0,     # set to zero if not considered ppl
            verbose=False,
            filter_cand=True,
            current_step=None
        ):

        main_device = self.models[0].device
        control_cands = []

        for j, worker in enumerate(self.workers):
            worker(self.prompts[j], "grad", worker.model)

        # Aggregate gradients
        grad = None
        for j, worker in enumerate(self.workers):
            new_grad = worker.results.get().to(main_device)
            new_grad = new_grad / new_grad.norm(dim=-1, keepdim=True)
            if grad is None:                    
                grad = torch.zeros_like(new_grad)
            if grad.shape != new_grad.shape:   
                with torch.no_grad():
                    control_cand = self.prompts[j-1].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
                    control_cands.append(self.get_filtered_cands(j-1, control_cand, filter_cand=filter_cand, curr_control=self.control_str))
                grad = new_grad
            else:
                grad += new_grad

        # grad is in shape length_control * vocabulary, 
        # e.g. in shape 20*32000
        with torch.no_grad():
            control_cand = self.prompts[j].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
            del grad, new_grad
            torch.cuda.empty_cache()

            control_cands.append(self.get_filtered_cands(j, control_cand, filter_cand=filter_cand, curr_control=self.control_str))

        del control_cand ; gc.collect()
        torch.cuda.empty_cache()

        # Search
        loss = torch.zeros(len(control_cands) * batch_size).to(main_device)             
        refusal_loss = torch.zeros(len(control_cands) * batch_size).to(main_device)     

        with torch.no_grad():
            for j, cand in enumerate(control_cands):

                progress = tqdm( range(len(self.prompts[0])), total=len(self.prompts[0]) ) if verbose else enumerate(self.prompts[0])
                for i in progress:

                    for k, worker in enumerate(self.workers):

                        worker(self.prompts[k][i], "logits", worker.model, test_controls = cand, return_ids=True)

                    logits, ids = zip(*[worker.results.get() for worker in self.workers])

                    torch.cuda.empty_cache()
                    if self.para.debug_mode:
                        print('-'*15 + 'some debug info'+'-'*15)
                        pass

                    temp_gcg_loss = sum([
                        target_weight * self.prompts[k][i].target_loss(logit, id).to(main_device) # may already go through cosine decay
                        for k, (logit, id) in enumerate(zip(logits, ids))
                    ])
                    loss[ j*batch_size : (j+1)*batch_size ] += temp_gcg_loss

                    if control_weight != 0:     
                        print("computing control weight!")
                        loss[j*batch_size:(j+1)*batch_size] += sum([
                            control_weight*self.prompts[k][i].control_loss(logit, id).mean(dim=-1).to(main_device)
                            for k, (logit, id) in enumerate(zip(logits, ids))
                        ])

                    if not self.para.use_augmented_loss:
                        overall_loss = loss
                    elif self.para.use_augmented_loss:    # DSN loss = affirmative loss + refusal loss
                        refusal_loss[ j*batch_size : (j+1)*batch_size ] += sum([
                            target_weight * self.prompts[k][i].dsn_refusal_loss(logit, id).to(main_device)
                            for k, (logit, id) in enumerate(zip(logits, ids))
                        ])
                        overall_loss = loss + refusal_loss

                    del logits, ids; gc.collect()
                    torch.cuda.empty_cache()

                    if verbose:                 
                        progress.set_description(f"loss={loss[j*batch_size:(j+1)*batch_size].min().item()/(i+1):.4f}")

            min_idx = overall_loss.argmin()
            model_idx = min_idx // batch_size
            batch_idx = min_idx % batch_size
            next_control, cand_loss = control_cands[model_idx][batch_idx], overall_loss[min_idx]

            loss_wrt_whole_ctrl = (overall_loss/len(self.prompts[0])/len(self.workers)).tolist()

        # to store two loss term during each step into a pth file
        logfile = self.para.result_prefix
        if logfile.endswith('.json'):
            loss_his_path = deepcopy(logfile)
            loss_his_path = loss_his_path.replace('.json', '_loss_history@step')

        if current_step == 1:
            store_gcg_loss_history_previous = []
            store_refusal_loss_history_previous = []
        else:
            temp_loss_dict = torch.load(loss_his_path+f"{current_step-1}.pth")
            store_gcg_loss_history_previous = temp_loss_dict['gcg']
            store_refusal_loss_history_previous = temp_loss_dict['refusal']

        store_gcg_loss_history = [loss[min_idx].item() / len(self.prompts[0]) / len(self.workers)]
        store_refusal_loss_history = [refusal_loss[min_idx].item() / len(self.prompts[0]) / len(self.workers)]
        dual_loss_his = {
            'gcg':store_gcg_loss_history_previous + store_gcg_loss_history,
            'refusal':store_refusal_loss_history_previous + store_refusal_loss_history
            }
        torch.save(dual_loss_his, loss_his_path+f"{current_step-1}.pth")
        os.rename(loss_his_path+f"{current_step-1}.pth", loss_his_path+f"{current_step}.pth")

        del loss, refusal_loss, overall_loss ; gc.collect()
        torch.cuda.empty_cache()

        tokenize_offset = 0
        if "Qwen2-7B-Instruct" in self.para.model_paths[0]:
            tokenize_offset = 1

        output_temp_str = 'Current length: '
        output_temp_str += str( len(self.workers[0].tokenizer(next_control).input_ids[1:])+tokenize_offset )
        output_temp_str += ' the adv suffix is now:'

        if self.logger is None:         # output by print
            print(output_temp_str)
            print(next_control)
        else:                           # output in the logger!
            self.logger.info(output_temp_str)
            self.logger.info(next_control)
        del output_temp_str

        return next_control, cand_loss.item() / len(self.prompts[0]) / len(self.workers), (control_cands[0], loss_wrt_whole_ctrl)