from research.secure_inference_3pc.backend import backend
from research.secure_inference_3pc.const import IGNORE_MSB_BITS
from research.secure_inference_3pc.modules.base import SecureModule
from research.secure_inference_3pc.conv2d.conv2d_handler_factory import conv2d_handler_factory
from research.secure_inference_3pc.modules.maxpool import SecureMaxPool
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER, MIN_VAL, MAX_VAL, SIGNED_DTYPE, \
    TRUNC_BITS
from research.bReLU import SecureOptimizedBlockReLU
from research.secure_inference_3pc.modules.base import Decompose
from research.secure_inference_3pc.const import  NUM_SPLIT_CONV_IN_CHANNEL, NUM_SPLIT_CONV_OUT_CHANNEL
from research.secure_inference_3pc.parties.server.numba_methods import private_compare_numba_server, post_compare_numba, \
    mult_server_numba

import numpy as np



class SecureConv2DServer(SecureModule):
    def __init__(self, W, bias, stride, dilation, padding, groups, **kwargs):
        super(SecureConv2DServer, self).__init__(**kwargs)

        self.W_plaintext = backend.put_on_device(W, self.device)
        self.bias = bias
        if self.bias is not None:
            self.bias = backend.reshape(self.bias, [1, -1, 1, 1])
            self.bias = backend.put_on_device(self.bias, self.device)
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups
        self.conv2d_handler = conv2d_handler_factory.create(self.device)
        self.is_dummy = False

        self.num_split_in_channels = NUM_SPLIT_CONV_IN_CHANNEL
        self.num_split_out_channels = NUM_SPLIT_CONV_OUT_CHANNEL

        self.out_channels, self.in_channels = self.W_plaintext.shape[:2]

        self.in_channel_group_size = self.in_channels // self.num_split_in_channels
        self.out_channel_group_size = self.out_channels // self.num_split_out_channels

        self.in_channel_s = [self.in_channel_group_size * i for i in range(self.num_split_in_channels)]
        self.in_channel_e = self.in_channel_s[1:] + [None]

        self.out_channel_s = [self.out_channel_group_size * i for i in range(self.num_split_out_channels)]
        self.out_channel_e = self.out_channel_s[1:] + [None]

    def split_conv(self, E_share, F_share, W_share, X_share):

        E_share_splits = [E_share[:, s_in:e_in] for s_in, e_in in zip(self.in_channel_s, self.in_channel_e)]
        F_share_splits = [[F_share[s_out:e_out, s_in:e_in] for s_in, e_in in zip(self.in_channel_s, self.in_channel_e)]
                          for s_out, e_out in zip(self.out_channel_s, self.out_channel_e)]
        X_share_splits = [X_share[:, s_in:e_in] for s_in, e_in in zip(self.in_channel_s, self.in_channel_e)]
        W_share_splits = [[W_share[s_out:e_out, s_in:e_in] for s_in, e_in in zip(self.in_channel_s, self.in_channel_e)]
                          for s_out, e_out in zip(self.out_channel_s, self.out_channel_e)]

        for i in range(self.num_split_in_channels):
            self.network_assets.sender_01.put(E_share_splits[i])
            for j in range(self.num_split_out_channels):
                self.network_assets.sender_01.put(F_share_splits[j][i])

        outs_all = []
        for i in range(self.num_split_in_channels):
            E_share_server = self.network_assets.receiver_01.get()
            E = E_share_server + E_share_splits[i]
            outs = []
            for j in range(self.num_split_out_channels):
                F_share_server = self.network_assets.receiver_01.get()
                F = F_share_server + F_share_splits[j][i]
                cur_W_share = W_share_splits[j][i] - F
                outs.append(self.conv2d_handler.conv2d(X_share_splits[i],
                                                       F,
                                                       E,
                                                       cur_W_share,
                                                       padding=self.padding,
                                                       stride=self.stride,
                                                       dilation=self.dilation,
                                                       groups=self.groups))
            outs_all.append(np.concatenate(outs, axis=1))
        outs_all = np.stack(outs_all).sum(axis=0)
        return outs_all
        # div = self.num_split_weights
        # mid = W_share.shape[0] // div
        # self.network_assets.sender_01.put(E_share)
        # E_share_client = self.network_assets.receiver_01.get()
        # E = backend.add(E_share_client, E_share, out=E_share)
        #
        # for i in range(div):
        #
        #     start = i * mid
        #     end = (i + 1) * mid if i < div else None
        #     self.network_assets.sender_01.put(F_share[start:end])
        #
        # outs = []
        # for i in range(div):
        #
        #     start = i * mid
        #     end = (i + 1) * mid if i < div else None
        #
        #     F_share_client = self.network_assets.receiver_01.get()
        #
        #     F = F_share_client + F_share[start:end]
        #
        #     W_share_cur = W_share[start:end] - F
        #
        #
        #     outs.append(self.conv2d_handler.conv2d(E,
        #                                            W_share_cur,
        #                                            X_share,
        #                                            F,
        #                                            padding=self.padding,
        #                                            stride=self.stride,
        #                                            dilation=self.dilation,
        #                                            groups=self.groups))
        # out = np.concatenate(outs, axis=1)
        # return out

    # @timer(name='server_conv2d')
    def forward(self, X_share):
        if self.is_dummy:
            X_share_client = self.network_assets.receiver_01.get()
            X = X_share_client + X_share
            out = self.conv2d_handler.conv2d(X, self.W_plaintext, None, None, self.padding, self.stride, self.dilation,
                                             self.groups)
            out = backend.right_shift(out, TRUNC_BITS, out=out)
            if self.bias is not None:
                out = backend.add(out, self.bias, out=out)
            mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=out.shape, dtype=SIGNED_DTYPE)
            return out - mu_0

        assert self.W_plaintext.shape[2] == self.W_plaintext.shape[3]
        assert (self.W_plaintext.shape[1] == X_share.shape[1]) or self.groups > 1
        assert self.stride[0] == self.stride[1]

        W_client = self.prf_handler[CLIENT, SERVER].integers(low=MIN_VAL, high=MAX_VAL, size=self.W_plaintext.shape,
                                                             dtype=SIGNED_DTYPE)
        A_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=X_share.shape,
                                                                     dtype=SIGNED_DTYPE)
        B_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=self.W_plaintext.shape,
                                                                     dtype=SIGNED_DTYPE)

        W_share = backend.subtract(self.W_plaintext, W_client, out=W_client)
        E_share = backend.subtract(X_share, A_share, out=A_share)
        F_share = backend.subtract(W_share, B_share, out=B_share)

        if self.num_split_out_channels == 1 and self.num_split_in_channels == 1:

            self.network_assets.sender_01.put(E_share)
            self.network_assets.sender_01.put(F_share)

            E_share_client = self.network_assets.receiver_01.get()
            F_share_client = self.network_assets.receiver_01.get()

            E = backend.add(E_share_client, E_share, out=E_share)
            F = backend.add(F_share_client, F_share, out=F_share)

            W_share = backend.subtract(W_share, F, out=W_share)

            out = self.conv2d_handler.conv2d(E, W_share, X_share, F, self.padding, self.stride, self.dilation,
                                             self.groups)

        else:
            out = self.split_conv(E_share, F_share, W_share, X_share)

        C_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=out.shape,
                                                                     dtype=SIGNED_DTYPE)

        out = backend.add(out, C_share, out=out)
        out = backend.right_shift(out, TRUNC_BITS, out=out)

        if self.bias is not None:
            out = backend.add(out, self.bias, out=out)

        mu_1 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=out.shape, dtype=SIGNED_DTYPE)
        mu_1 = backend.multiply(mu_1, -1, out=mu_1)
        out = backend.add(out, mu_1, out=out)

        return out


class PrivateCompareServer(SecureModule):
    def __init__(self, **kwargs):
        super(PrivateCompareServer, self).__init__(**kwargs)
        # self.decompose = Decompose(ignore_msb_bits=IGNORE_MSB_BITS, num_of_compare_bits=64,
        #                            dtype=SIGNED_DTYPE, **kwargs)

    def forward(self, x_bits_1, r, beta):
        s = self.prf_handler[CLIENT, SERVER].integers(low=1, high=67, size=x_bits_1.shape, dtype=backend.int8)
        d_bits_1 = private_compare_numba_server(s, r, x_bits_1, beta, IGNORE_MSB_BITS)

        # r[backend.astype(beta, backend.bool)] += 1
        # bits = self.decompose(r)
        # c_bits_1 = get_c_party_1(x_bits_1, bits, beta)
        # s = backend.multiply(s, c_bits_1, out=s)
        # d_bits_1 = module_67(s)

        d_bits_1 = self.prf_handler[CLIENT, SERVER].permutation(d_bits_1, axis=-1)

        self.network_assets.sender_12.put(d_bits_1)

        return


class ShareConvertServer(SecureModule):
    def __init__(self, **kwargs):
        super(ShareConvertServer, self).__init__(**kwargs)
        self.private_compare = PrivateCompareServer(**kwargs)

    def forward(self, a_1):
        eta_pp = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_1.shape, dtype=backend.int8)
        r = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=a_1.shape, dtype=SIGNED_DTYPE)
        r_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=a_1.shape, dtype=SIGNED_DTYPE)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=a_1.shape, dtype=SIGNED_DTYPE)

        r_1 = backend.subtract(r, r_0, out=r_0)
        a_tild_1 = backend.add(a_1, r_1, out=r_1)
        beta_1 = backend.astype(0 < a_1 - a_tild_1, SIGNED_DTYPE)  # TODO: Optimize this

        self.network_assets.sender_12.put(a_tild_1)

        x_bits_1 = self.network_assets.receiver_12.get()

        delta_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=a_1.shape,
                                                                     dtype=SIGNED_DTYPE)

        r_minus_1 = backend.subtract(r, 1, out=r)
        self.private_compare(x_bits_1, r_minus_1, eta_pp)
        eta_p_1 = self.network_assets.receiver_12.get()

        return post_compare_numba(a_1, eta_pp, delta_1, beta_1, mu_0, eta_p_1)
        # mu_1 = backend.multiply(mu_0, -1, out=mu_0)
        # eta_pp = backend.astype(eta_pp, SIGNED_DTYPE)
        # t00 = backend.multiply(eta_pp, eta_p_1, out=eta_pp)
        # t11 = self.add_mode_L_minus_one(t00, t00)
        # eta_1 = self.sub_mode_L_minus_one(eta_p_1, t11)
        # t00 = self.add_mode_L_minus_one(delta_1, eta_1)
        # theta_1 = self.add_mode_L_minus_one(beta_1, t00)
        # y_1 = self.sub_mode_L_minus_one(a_1, theta_1)
        # y_1 = self.add_mode_L_minus_one(y_1, mu_1)
        # return y_1


from torch.nn import Module


class DepthToSpace(Module):

    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def forward(self, x):
        N, C, H, W, _ = x.shape
        x = x.reshape(N, C, H, W, self.block_size[0], self.block_size[1])
        x = backend.permute(x, (0, 1, 2, 4, 3, 5))
        x = x.reshape(N, C, H * self.block_size[0], W * self.block_size[1])
        return x


class SecurePostBReLUMultServer(SecureModule):
    def __init__(self, **kwargs):
        super(SecurePostBReLUMultServer, self).__init__(**kwargs)

    def post(self, activation, sign_tensors, cumsum_shapes, pad_handlers, active_block_sizes,
             active_block_sizes_to_channels):
        relu_map = backend.ones_like(activation)
        for i, block_size in enumerate(active_block_sizes):
            orig_shape = (1, active_block_sizes_to_channels[i].shape[0], pad_handlers[i].out_shape[0] // block_size[0],
                          pad_handlers[i].out_shape[1] // block_size[1], 1)
            sign_tensor = sign_tensors[int(cumsum_shapes[i]):int(cumsum_shapes[i + 1])].reshape(orig_shape)
            tensor = backend.repeat(sign_tensor, block_size[0] * block_size[1])
            cur_channels = active_block_sizes_to_channels[i]
            relu_map[:, cur_channels] = pad_handlers[i].unpad(DepthToSpace(active_block_sizes[i])(tensor))
        return relu_map

    def forward(self, activation, sign_tensors, cumsum_shapes, pad_handlers, is_identity_channels, active_block_sizes,
                active_block_sizes_to_channels):
        non_identity_activation = activation[:, ~is_identity_channels]

        A_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1,
                                                                     size=non_identity_activation.shape,
                                                                     dtype=SIGNED_DTYPE)
        B_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=sign_tensors.shape,
                                                                     dtype=SIGNED_DTYPE)
        C_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1,
                                                                     size=non_identity_activation.shape,
                                                                     dtype=SIGNED_DTYPE)
        mu_1 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=activation.shape,
                                                         dtype=activation.dtype)

        E_share = backend.subtract(non_identity_activation, A_share, out=A_share)
        F_share = backend.subtract(sign_tensors, B_share, out=B_share)

        self.network_assets.sender_01.put(E_share)
        E_share_client = self.network_assets.receiver_01.get()
        self.network_assets.sender_01.put(F_share)
        F_share_client = self.network_assets.receiver_01.get()

        E = backend.add(E_share_client, E_share, out=E_share)
        F = backend.add(F_share_client, F_share, out=F_share)

        F = self.post(activation, F, cumsum_shapes, pad_handlers, active_block_sizes, active_block_sizes_to_channels)[:,
            ~is_identity_channels]
        sign_tensors = self.post(activation, sign_tensors, cumsum_shapes, pad_handlers, active_block_sizes,
                                 active_block_sizes_to_channels)[:, ~is_identity_channels]

        out = - E * F + non_identity_activation * F + sign_tensors * E + C_share
        activation[:, ~is_identity_channels] = out
        mu_1 = backend.multiply(mu_1, -1, out=mu_1)
        activation = activation + mu_1
        return activation


class SecureMultiplicationServer(SecureModule):
    def __init__(self, **kwargs):
        super(SecureMultiplicationServer, self).__init__(**kwargs)

    def forward(self, X_share, Y_share):
        A_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape,
                                                                     dtype=SIGNED_DTYPE)
        B_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape,
                                                                     dtype=SIGNED_DTYPE)
        C_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape,
                                                                     dtype=SIGNED_DTYPE)
        mu_1 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=X_share.shape, dtype=X_share.dtype)

        E_share = backend.subtract(X_share, A_share, out=A_share)
        F_share = backend.subtract(Y_share, B_share, out=B_share)

        self.network_assets.sender_01.put(E_share)
        E_share_client = self.network_assets.receiver_01.get()
        self.network_assets.sender_01.put(F_share)
        F_share_client = self.network_assets.receiver_01.get()

        out = mult_server_numba(X_share, Y_share, C_share, mu_1, E_share, E_share_client, F_share, F_share_client)

        # E = backend.add(E_share_client, E_share, out=E_share)
        # F = backend.add(F_share_client, F_share, out=F_share)
        # out = - E * F + X_share * F + Y_share * E + C_share
        # mu_1 = backend.multiply(mu_1, -1, out=mu_1)
        # out = out + mu_1

        return out


class SecureMSBServer(SecureModule):
    def __init__(self, **kwargs):
        super(SecureMSBServer, self).__init__(**kwargs)
        self.mult = SecureMultiplicationServer(**kwargs)
        self.private_compare = PrivateCompareServer(**kwargs)

    def forward(self, a_1):
        beta = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_1.shape, dtype=backend.int8)
        x_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=a_1.shape, dtype=SIGNED_DTYPE)
        mu_1 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=a_1.shape, dtype=a_1.dtype)
        mu_1 = backend.multiply(mu_1, -1, out=mu_1)

        x_bits_1 = self.network_assets.receiver_12.get()
        x_bit_0_1 = self.network_assets.receiver_12.get()

        y_1 = self.add_mode_L_minus_one(a_1, a_1)
        r_1 = self.add_mode_L_minus_one(x_1, y_1)

        self.network_assets.sender_01.put(r_1)
        r_0 = self.network_assets.receiver_01.get()

        r = self.add_mode_L_minus_one(r_0, r_1)
        r_mod_2 = r % 2

        self.private_compare(x_bits_1, r, beta)
        beta_p_1 = self.network_assets.receiver_12.get()

        beta = backend.astype(beta, SIGNED_DTYPE)  # TODO: Optimize this
        gamma_1 = beta_p_1 + beta - 2 * beta * beta_p_1  # TODO: Optimize this
        delta_1 = x_bit_0_1 + r_mod_2 - (2 * r_mod_2 * x_bit_0_1)  # TODO: Optimize this

        theta_1 = self.mult(gamma_1, delta_1)

        alpha_1 = gamma_1 + delta_1 - 2 * theta_1  # TODO: Optimize this
        alpha_1 = alpha_1 + mu_1  # TODO: Optimize this

        return alpha_1


class SecureDReLUServer(SecureModule):
    # counter = 0
    def __init__(self, **kwargs):
        super(SecureDReLUServer, self).__init__(**kwargs)

        self.share_convert = ShareConvertServer(**kwargs)
        self.msb = SecureMSBServer(**kwargs)

    def forward(self, X_share):
        # SecureDReLUServer.counter += 1
        # np.save("/home/yakir/Data2/secure_activation_statistics/server/{}.npy".format(SecureDReLUServer.counter), X_share)

        mu_1 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape, dtype=SIGNED_DTYPE)
        backend.multiply(mu_1, -1, out=mu_1)

        X1_converted = self.share_convert(X_share)
        MSB_1 = self.msb(X1_converted)

        ret = backend.multiply(MSB_1, -1, out=MSB_1)
        ret = backend.add(ret, mu_1, out=ret)
        ret = backend.add(ret, 1, out=ret)
        return ret


class SecureReLUServer(SecureModule):
    # index = 0
    def __init__(self, dummy_relu=False, **kwargs):
        super(SecureReLUServer, self).__init__(**kwargs)

        self.DReLU = SecureDReLUServer(**kwargs)
        self.mult = SecureMultiplicationServer(**kwargs)
        self.dummy_relu = dummy_relu

    def forward(self, X_share):
        # return X_share
        if self.dummy_relu:
            share_client = self.network_assets.receiver_01.get()
            recon = share_client + X_share
            value = recon * (backend.astype(recon > 0, recon.dtype))
            mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=value.shape, dtype=SIGNED_DTYPE)
            return value - mu_0
        else:

            shape = X_share.shape
            mu_1 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
            backend.multiply(mu_1, -1, out=mu_1)

            X_share = X_share.reshape(-1)
            MSB_0 = self.DReLU(X_share)
            ret = self.mult(X_share, MSB_0).reshape(shape)
            backend.add(ret, mu_1, out=ret)
            return ret


class SecureBlockReLUServer(SecureModule, SecureOptimizedBlockReLU):
    def __init__(self, block_sizes, dummy_relu=False, **kwargs):
        SecureModule.__init__(self, **kwargs)
        SecureOptimizedBlockReLU.__init__(self, block_sizes)
        self.DReLU = SecureDReLUServer(**kwargs)
        self.mult = SecureMultiplicationServer(**kwargs)
        self.dummy_relu = dummy_relu
        self.post_bReLU = SecurePostBReLUMultServer(**kwargs)

    def forward(self, activation):
        if self.dummy_relu:
            return activation
        return SecureOptimizedBlockReLU.forward(self, activation)


class SecureSelectShareServer(SecureModule):
    def __init__(self, **kwargs):
        super(SecureSelectShareServer, self).__init__(**kwargs)
        self.secure_multiplication = SecureMultiplicationServer(**kwargs)

    def forward(self, alpha, x, y):
        mu_1 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=alpha.shape, dtype=SIGNED_DTYPE)
        mu_1 = backend.multiply(mu_1, -1, out=mu_1)
        y = backend.subtract(y, x, out=y)

        c = self.secure_multiplication(alpha, y)
        x = backend.add(x, c, out=x)
        x = backend.add(x, mu_1, out=x)
        return x


class SecureMaxPoolServer(SecureMaxPool):
    def __init__(self, kernel_size=3, stride=2, padding=1, **kwargs):
        super(SecureMaxPoolServer, self).__init__(kernel_size, stride, padding, **kwargs)
        self.select_share = SecureSelectShareServer(**kwargs)
        self.dReLU = SecureDReLUServer(**kwargs)
        self.mult = SecureMultiplicationServer(**kwargs)

    def forward(self, x):
        ret = super(SecureMaxPoolServer, self).forward(x)
        mu_1 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=ret.shape, dtype=SIGNED_DTYPE)
        mu_1 = backend.multiply(mu_1, -1, out=mu_1)
        ret = backend.add(ret, mu_1, out=ret)
        return ret
