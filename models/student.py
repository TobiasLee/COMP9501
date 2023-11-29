import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import BertForSequenceClassification


def get_sorted_indices(selection_strategy, logits, labels, selection_ratio):
    bsz = logits.size(0)
    device = logits.device
    if selection_strategy == "none":
        indices = torch.arange(logits.size(0), device=device)
    elif selection_strategy == "entropy":
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs), dim=1)
        _, indices = torch.sort(entropy, descending=True)
    elif selection_strategy == "entropy-r":
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(
            probs * torch.log(probs), dim=1
        )  # select most certain data
        _, indices = torch.sort(entropy, descending=False)
    elif (
        selection_strategy == "kl" or selection_strategy == "kl-fix"
    ):  # kl between logits & labels
        probs = F.log_softmax(logits, dim=-1)
        kl_distance = F.kl_div(probs, labels, reduction="none").sum(dim=-1)  # bsz,
        _, indices = torch.sort(kl_distance, descending=True)
    elif selection_strategy == "kl-b":  # balanced difficulty & easy
        probs = F.log_softmax(logits, dim=-1)
        kl_distance = F.kl_div(probs, labels, reduction="none").sum(dim=-1)  # bsz,
        _, indices_d2e = torch.sort(
            kl_distance, descending=True
        )  # difficult to easy indices
        _, indices_e2d = torch.sort(
            kl_distance, descending=False
        )  # easy to difficult indices
        interleave = torch.stack((indices_d2e, indices_e2d), dim=1)
        indices = interleave.view(-1, 1).squeeze()[:bsz]  # one difficult one easy ...
    elif selection_strategy == "kl-r":  # kl between logits & labels
        probs = F.log_softmax(logits, dim=-1)
        kl_distance = F.kl_div(probs, labels, reduction="none").sum(dim=-1)  # bsz,
        _, indices = torch.sort(kl_distance, descending=False)  # select most easy data?
    elif (
        selection_strategy == "random"
    ):  # use random selected mixup samples for training
        indices = torch.randperm(bsz, device=device)
    elif selection_strategy == "confidence":
        s_probs = F.softmax(logits, dim=-1)
        s_conf, _ = torch.max(s_probs, dim=-1)
        _, indices = torch.sort(
            s_conf, descending=False
        )  # lower confidence indicate challenging input
    elif selection_strategy == "confidence-r":
        s_probs = F.softmax(logits, dim=-1)
        s_conf, _ = torch.max(s_probs, dim=-1)
        _, indices = torch.sort(
            s_conf, descending=True
        )  # lower confidence indicate challenging input
    elif selection_strategy == "margin":
        s_probs = F.softmax(logits, dim=-1)
        sorted_probs, _ = torch.sort(s_probs, dim=-1, descending=True)
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]  # top-1 prob - top-2 prob
        _, indices = torch.sort(
            margin, descending=False
        )  # lower margin indicate more uncertain examples
    else:
        raise ValueError("Unsupported uncertainty strategy")
    if selection_strategy != "none":
        indices = indices[: int(bsz * selection_ratio)]
    # print(indices.size())
    return indices


class DynamicDataKDForSequenceClassification(BertForSequenceClassification):
    def __init__(
        self,
        config,
        kd_alpha=1.0,
        ce_alpha=1.0,
        teacher=None,
        temperature=5.0,
        kl_kd=False,
        selection_strategy="none",
        selection_ratio=1.0,
    ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.teacher = teacher
        self.kd_alpha = kd_alpha
        self.ce_alpha = ce_alpha
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.kl_kd = kl_kd
        self.temperature = temperature
        self.selection_strategy = selection_strategy
        self.selection_ratio = selection_ratio

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        kd_loss = None

        if self.training:
            assert self.teacher is not None, "student hold a None teacher reference"
            student_outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            pooled_output = student_outputs[1]
            pooled_output = self.dropout(pooled_output)
            student_logits = self.classifier(pooled_output)

            indices = get_sorted_indices(
                self.selection_strategy, student_logits, labels, self.selection_ratio
            )

            with torch.no_grad():
                teacher_outputs = self.teacher(
                    input_ids[indices],
                    attention_mask=attention_mask[indices]
                    if attention_mask is not None
                    else None,
                    token_type_ids=token_type_ids[indices]
                    if token_type_ids is not None
                    else None,
                    position_ids=position_ids[indices]
                    if position_ids is not None
                    else None,
                    head_mask=head_mask[indices] if head_mask is not None else None,
                    inputs_embeds=inputs_embeds[indices]
                    if inputs_embeds is not None
                    else None,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                teacher_logits = teacher_outputs[0]

            student_logits_for_kd = student_logits[indices]
            if self.kl_kd:
                kd_loss = (
                    self.kl_loss(
                        F.log_softmax(student_logits_for_kd / self.temperature, dim=1),
                        F.softmax(teacher_logits / self.temperature, dim=1),
                    )
                    * self.temperature**2
                )
            else:
                kd_loss = self.mse_loss(student_logits_for_kd, teacher_logits)

        else:  # use student model for inference
            student_outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            pooled_output = student_outputs[1]
            pooled_output = self.dropout(pooled_output)
            student_logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = self.ce_alpha * loss_fct(
                    student_logits.view(-1), labels.view(-1)
                )
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = self.ce_alpha * loss_fct(
                    student_logits.view(-1, self.num_labels), labels.view(-1)
                )

            if kd_loss is not None:
                loss += self.kd_alpha * kd_loss

        output = (student_logits,) + student_outputs[2:]
        return ((loss,) + output) if loss is not None else output
