import torch
import torch.nn as nn
from base import BaseModel
from utils.losses import *
from models.decoders import *
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, feature_base=32, drop_rate=0):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, feature_base, kernel_size=3, stride=1, padding=(1, 1), bias=False) # not sure of stride here
        self.dropout1 = nn.Dropout(drop_rate)
        self.res_block1 = Res_Block(feature_base, drop_rate)
        self.pool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(feature_base, feature_base*2, kernel_size=3, stride=1, padding=(1, 1), bias=False)  # not sure of stride here
        self.dropout2 = nn.Dropout(drop_rate)
        self.res_block2 = Res_Block(feature_base*2, drop_rate)
        self.pool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(feature_base*2, feature_base * 4, kernel_size=3, stride=1, padding=(1, 1), bias=False)  # not sure of stride here
        self.dropout3 = nn.Dropout(drop_rate)
        self.res_block3 = Res_Block(feature_base * 4, drop_rate)
        self.pool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv4 = nn.Conv2d(feature_base * 4, feature_base * 8, kernel_size=3, stride=1, padding=(1, 1),
                               bias=False)  # not sure of stride here
        self.dropout4 = nn.Dropout(drop_rate)
        self.res_block4 = Res_Block(feature_base * 8, drop_rate)
        self.pool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv5 = nn.Conv2d(feature_base * 8, feature_base * 16, kernel_size=3, stride=1, padding=(1, 1),
                               bias=False)  # not sure of stride here
        self.dropout5 = nn.Dropout(drop_rate)
        self.res_block5_1 = Res_Block(feature_base * 16, drop_rate)
        self.res_block5_2 = Res_Block(feature_base * 16, drop_rate)

    def forward(self, x):
        res1 = self.res_block1(self.dropout1(self.conv1(x)))
        res2 = self.res_block2(self.dropout2(self.conv2(self.pool1(res1))))
        res3 = self.res_block3(self.dropout3(self.conv3(self.pool2(res2))))
        res4 = self.res_block4(self.dropout4(self.conv4(self.pool3(res3))))
        res5_1 = self.res_block5_1(self.dropout5(self.conv5(self.pool4(res4))))
        res5_2 = self.res_block5_2(res5_1)
        return [res1, res2, res3, res4, res5_2]


class Res_Block(nn.Module):
    def __init__(self, nb_channels, drop_rate):
        super(Res_Block, self).__init__()
        self.bn = nn.BatchNorm2d(nb_channels)  # to determine
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        inner_conv1 = self.dropout(self.conv(self.relu(self.bn(x))))
        inner_conv2 = self.dropout(self.conv(self.relu(self.bn(inner_conv1))))
        return inner_conv2 + x


class Decoder(nn.Module):
    def __init__(self, num_classes, feature_base=32, drop_rate=0):
        super(Decoder, self).__init__()
        self.deconv6 = Deconv(feature_base*16, feature_base*8, drop_rate)
        self.conv6 = nn.Conv2d(feature_base*16, feature_base*8, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.res_block6 = Res_Block(feature_base * 8, drop_rate)
        self.deconv7 = Deconv(feature_base*8, feature_base*4, drop_rate)
        self.conv7 = nn.Conv2d(feature_base * 8, feature_base * 4, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.res_block7 = Res_Block(feature_base * 4, drop_rate)
        self.deconv8 = Deconv(feature_base*4, feature_base*2, drop_rate)
        self.conv8 = nn.Conv2d(feature_base * 4, feature_base * 2, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.res_block8 = Res_Block(feature_base * 2, drop_rate)
        self.deconv9 = Deconv(feature_base*2, feature_base, drop_rate)
        self.conv9 = nn.Conv2d(feature_base*2, feature_base, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.res_block9 = Res_Block(feature_base, drop_rate)
        self.bn = nn.BatchNorm2d(feature_base)  # to determine
        self.relu = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(feature_base, num_classes, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        res1, res2, res3, res4, res5_2 = x
        sum6 = torch.cat([res4, self.deconv6(res5_2)], dim=1)
        res6 = self.res_block6(self.conv6(sum6))
        sum7 = torch.cat([res3, self.deconv7(res6)], dim=1)
        res7 = self.res_block7(self.conv7(sum7))
        sum8 = torch.cat([res2, self.deconv8(res7)], dim=1)
        res8 = self.res_block8(self.conv8(sum8))
        sum9 = torch.cat([res1, self.deconv9(res8)], dim=1)
        res9 = self.res_block9(self.conv9(sum9))
        output = self.dropout(self.conv10(self.relu(self.bn(res9))))

        return output


class Deconv(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate):
        super(Deconv, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)  # to determine
        self.relu = nn.ReLU(inplace=True)
        self.convT = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=(1, 1),
                                        output_padding=1, bias=False)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        deconv = self.dropout(self.convT(self.relu(self.bn(x))))
        return deconv


class DropOutDecoder(nn.Module):
    def __init__(self, num_classes, drop_rate=0.3, spatial_dropout=True):
        super(DropOutDecoder, self).__init__()
        self.dropout = nn.Dropout2d(p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)
        self.decoder = Decoder(num_classes)

    def forward(self, x,_):
        res1, res2, res3, res4, res5_2 = x
        res5_2 = self.dropout(res5_2)
        x = [res1, res2, res3, res4, res5_2]
        x = self.decoder(x)
        return x


class FeatureDropDecoder(nn.Module):
    def __init__(self, num_classes):
        super(FeatureDropDecoder, self).__init__()
        self.decoder = Decoder(num_classes)

    def feature_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(0.7, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)

    def forward(self, x,_):
        res1, res2, res3, res4, res5_2 = x
        res5_2 = self.feature_dropout(res5_2)
        x = [res1, res2, res3, res4, res5_2]
        x = self.decoder(x)
        return x


class FeatureNoiseDecoder(nn.Module):
    def __init__(self, num_classes, uniform_range=0.3):
        super(FeatureNoiseDecoder, self).__init__()
        self.decoder = Decoder(num_classes)
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x,_):
        res1, res2, res3, res4, res5_2 = x
        res5_2 = self.feature_based_noise(res5_2)
        x = [res1, res2, res3, res4, res5_2]
        x = self.decoder(x)
        return x


def _l2_normalize(d):
    # Normalizing per batch axis
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def get_r_adv(x, decoder, it=1, xi=1e-1, eps=10.0):
    """
    Virtual Adversarial Training
    https://arxiv.org/abs/1704.03976
    """
    res1, res2, res3, res4, res5_2 = x
    res1_detached = res1.detach()
    res2_detached = res2.detach()
    res3_detached = res3.detach()
    res4_detached = res4.detach()
    res5_2_detached = res5_2.detach()
    x_detached = [res1_detached, res2_detached, res3_detached, res4_detached, res5_2_detached]
    with torch.no_grad():
        pred = F.softmax(decoder(x_detached), dim=1)

    d = torch.rand(res5_2.shape).sub(0.5).to(res5_2.device)
    d = _l2_normalize(d)

    for _ in range(it):
        d.requires_grad_()
        x_detached[4] = x_detached[4] + xi * d
        pred_hat = decoder(x_detached)
        logp_hat = F.log_softmax(pred_hat, dim=1)
        adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
        adv_distance.backward()
        d = _l2_normalize(d.grad)
        decoder.zero_grad()

    r_adv = d * eps
    return r_adv


class VATDecoder(nn.Module):
    def __init__(self, num_classes, xi=1e-1, eps=10.0, iterations=1):
        super(VATDecoder, self).__init__()
        self.xi = xi
        self.eps = eps
        self.it = iterations
        self.decoder = Decoder(num_classes)

    def forward(self, x,_):
        r_adv = get_r_adv(x, self.decoder, self.it, self.xi, self.eps)
        res1, res2, res3, res4, res5_2 = x
        x = [res1, res2, res3, res4, res5_2 + r_adv]
        x = self.decoder(x)
        return x


def guided_cutout(output, resize, erase=0.4, use_dropout=False):
    if len(output.shape) == 3:
        masks = (output > 0).float()
    else:
        masks = (output.argmax(1) > 0).float()

    if use_dropout:
        p_drop = random.randint(3, 6) / 10
        maskdroped = (F.dropout(masks, p_drop) > 0).float()
        maskdroped = maskdroped + (1 - masks)
        maskdroped.unsqueeze_(0)
        maskdroped = F.interpolate(maskdroped, size=resize, mode='nearest')

    masks_np = []
    for mask in masks:
        mask_np = np.uint8(mask.cpu().numpy())
        mask_ones = np.ones_like(mask_np)
        try:  # Version 3.x
            _, contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except:  # Version 4.x
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polys = [c.reshape(c.shape[0], c.shape[-1]) for c in contours if c.shape[0] > 50]
        for poly in polys:
            min_w, max_w = poly[:, 0].min(), poly[:, 0].max()
            min_h, max_h = poly[:, 1].min(), poly[:, 1].max()
            bb_w, bb_h = max_w - min_w, max_h - min_h
            rnd_start_w = random.randint(0, int(bb_w * (1 - erase)))
            rnd_start_h = random.randint(0, int(bb_h * (1 - erase)))
            h_start, h_end = min_h + rnd_start_h, min_h + rnd_start_h + int(bb_h * erase)
            w_start, w_end = min_w + rnd_start_w, min_w + rnd_start_w + int(bb_w * erase)
            mask_ones[h_start:h_end, w_start:w_end] = 0
        masks_np.append(mask_ones)
    masks_np = np.stack(masks_np)

    maskcut = torch.from_numpy(masks_np).float().unsqueeze_(1)
    maskcut = F.interpolate(maskcut, size=resize, mode='nearest')

    if use_dropout:
        return maskcut.to(output.device), maskdroped.to(output.device)
    return maskcut.to(output.device)


class CutOutDecoder(nn.Module):
    def __init__(self, num_classes, drop_rate=0.3, spatial_dropout=True, erase=0.4):
        super(CutOutDecoder, self).__init__()
        self.erase = erase
        self.decoder = Decoder(num_classes)

    def forward(self, x, pred=None):
        res1, res2, res3, res4, res5_2 = x
        maskcut = guided_cutout(pred, erase=self.erase, resize=(res5_2.size(2), res5_2.size(3)))
        res1, res2, res3, res4, res5_2 = x
        res5_2 = res5_2 * maskcut
        x = [res1, res2, res3, res4, res5_2]
        x = self.decoder(x)
        return x


def guided_masking(x, output, resize, return_msk_context=True):
    if len(output.shape) == 3:
        masks_context = (output > 0).float().unsqueeze(1)
    else:
        masks_context = (output.argmax(1) > 0).float().unsqueeze(1)

    masks_context = F.interpolate(masks_context, size=resize, mode='nearest')

    res1, res2, res3, res4, res5_2 = x
    res5_2 = masks_context * res5_2
    if return_msk_context:
        return [res1, res2, res3, res4, res5_2]

    masks_objects = (1 - masks_context)
    res5_2 = masks_objects * res5_2
    return [res1, res2, res3, res4, res5_2]


class ContextMaskingDecoder(nn.Module):
    def __init__(self, num_classes):
        super(ContextMaskingDecoder, self).__init__()
        self.decoder = Decoder(num_classes)

    def forward(self, x, pred=None):
        res1, res2, res3, res4, res5_2 = x
        x_masked_context = guided_masking(x, pred, resize=(res5_2.size(2), res5_2.size(3)),
                                           return_msk_context=True)
        x_masked_context = self.decoder(x_masked_context)
        return x_masked_context


class ObjectMaskingDecoder(nn.Module):
    def __init__(self, num_classes):
        super(ObjectMaskingDecoder, self).__init__()
        self.decoder = Decoder(num_classes)

    def forward(self, x, pred=None):
        res1, res2, res3, res4, res5_2 = x
        x_masked_obj = guided_masking(x, pred, resize=(res5_2.size(2), res5_2.size(3)), return_msk_context=False)
        x_masked_obj = self.decoder(x_masked_obj)

        return x_masked_obj

class CCT_Unet(BaseModel):
    def __init__(self, encoder, num_classes, conf, sup_loss=None, cons_w_unsup=None, testing=False,
                 use_weak_lables=False, weakly_loss_w=0.4):
        if not testing:
            assert (sup_loss is not None) and (cons_w_unsup is not None)

        super(CCT_Unet, self).__init__()
        assert int(conf['supervised']) + int(conf['semi']) == 1, 'one mode only'
        if conf['supervised']:
            self.mode = 'supervised'
        else:
            self.mode = 'semi'

        # Supervised and unsupervised losses
        if conf['un_loss'] == "KL":
            self.unsuper_loss = softmax_kl_loss
        elif conf['un_loss'] == "MSE":
            self.unsuper_loss = softmax_mse_loss
        elif conf['un_loss'] == "JS":
            self.unsuper_loss = softmax_js_loss
        else:
            raise ValueError(f"Invalid supervised loss {conf['un_loss']}")

        self.unsup_loss_w = cons_w_unsup
        self.sup_loss_w = conf['supervised_w']
        self.softmax_temp = conf['softmax_temp']
        self.sup_loss = sup_loss
        self.sup_type = conf['sup_loss']

        # Use weak labels
        self.use_weak_lables = use_weak_lables
        self.weakly_loss_w = weakly_loss_w
        # pair wise loss (sup mat)
        self.aux_constraint = conf['aux_constraint']
        self.aux_constraint_w = conf['aux_constraint_w']
        # confidence masking (sup mat)
        self.confidence_th = conf['confidence_th']
        self.confidence_masking = conf['confidence_masking']

        # Create the model
        self.encoder = encoder
        self.decoder = Decoder(num_classes)

        # The auxilary decoders
        if self.mode == 'semi' or self.mode == 'weakly_semi':
            vat_decoder = [VATDecoder(num_classes, xi=conf['xi'], eps=conf['eps']) for _ in range(conf['vat'])]
            drop_decoder = [DropOutDecoder(num_classes, drop_rate=conf['drop_rate'], spatial_dropout=conf['spatial'])
                            for _ in range(conf['drop'])]
            cut_decoder = [CutOutDecoder(num_classes, erase=conf['erase']) for _ in range(conf['cutout'])]
            context_m_decoder = [ContextMaskingDecoder(num_classes) for _ in range(conf['context_masking'])]
            object_masking = [ObjectMaskingDecoder(num_classes) for _ in range(conf['object_masking'])]
            feature_drop = [FeatureDropDecoder(num_classes) for _ in range(conf['feature_drop'])]
            feature_noise = [FeatureNoiseDecoder(num_classes, uniform_range=conf['uniform_range'])
                             for _ in range(conf['feature_noise'])]

            self.aux_decoders = nn.ModuleList([*vat_decoder, *drop_decoder, *cut_decoder,
                                               *context_m_decoder, *object_masking, *feature_drop, *feature_noise])

    def forward(self, x_l=None, target_l=None, x_ul=None, target_ul=None, curr_iter=None, epoch=None):
        if not self.training:
            return self.decoder(self.encoder(x_l))

        # We compute the losses in the forward pass to avoid problems encountered in muti-gpu

        # Forward pass the labels example
        input_size = (x_l.size(2), x_l.size(3))
        z_l = self.encoder(x_l)
        output_l = self.decoder(z_l)
        if output_l.shape != x_l.shape:
            output_l = F.interpolate(output_l, size=input_size, mode='bilinear', align_corners=True)

        # Supervised loss
        if self.sup_type == 'CE':
            loss_sup = self.sup_loss(output_l, target_l, temperature=self.softmax_temp) * self.sup_loss_w
        else:
            loss_sup = self.sup_loss(output_l, target_l, curr_iter=curr_iter, epoch=epoch) * self.sup_loss_w

        # If supervised mode only, return
        if self.mode == 'supervised':
            curr_losses = {'loss_sup': loss_sup}
            outputs = {'sup_pred': output_l}
            total_loss = loss_sup
            return total_loss, curr_losses, outputs

            # If semi supervised mode
        elif self.mode == 'semi':
            # Get main prediction
            x_ul = self.encoder(x_ul)
            output_ul = self.decoder(x_ul)

            # Get auxiliary predictions
            outputs_ul = [aux_decoder(x_ul, output_ul.detach()) for aux_decoder in self.aux_decoders]
            targets = F.softmax(output_ul.detach(), dim=1)

            # Compute unsupervised loss
            loss_unsup = sum([self.unsuper_loss(inputs=u, targets=targets, \
                                                conf_mask=self.confidence_masking, threshold=self.confidence_th,
                                                use_softmax=False)
                              for u in outputs_ul])
            loss_unsup = (loss_unsup / len(outputs_ul))
            curr_losses = {'loss_sup': loss_sup}

            if output_ul.shape != x_l.shape:
                output_ul = F.interpolate(output_ul, size=input_size, mode='bilinear', align_corners=True)
            outputs = {'sup_pred': output_l, 'unsup_pred': output_ul}

            # Compute the unsupervised loss
            weight_u = self.unsup_loss_w(epoch=epoch, curr_iter=curr_iter)
            loss_unsup = loss_unsup * weight_u
            curr_losses['loss_unsup'] = loss_unsup
            total_loss = loss_unsup + loss_sup

            return total_loss, curr_losses, outputs

    def get_other_params(self):
        if self.mode == 'semi':
            return chain(self.encoder.parameters(), self.decoder.parameters(),
                         self.aux_decoders.parameters())

        return chain(self.encoder.parameters(), self.decoder.parameters())
