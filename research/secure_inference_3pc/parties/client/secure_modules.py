from research.secure_inference_3pc.backend import backend
from research.secure_inference_3pc.modules.base import SecureModule
from research.secure_inference_3pc.base import P
from research.secure_inference_3pc.conv2d.conv2d_handler_factory import conv2d_handler_factory
from research.secure_inference_3pc.modules.maxpool import SecureMaxPool
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER, MIN_VAL, MAX_VAL, SIGNED_DTYPE, IGNORE_MSB_BITS, TRUNC_BITS
from research.secure_inference_3pc.conv2d.utils import get_output_shape
from research.secure_inference_3pc.modules.base import Decompose
from research.bReLU import SecureOptimizedBlockReLU
from research.secure_inference_3pc.const import NUM_SPLIT_CONV_IN_CHANNEL, NUM_SPLIT_CONV_OUT_CHANNEL
from research.secure_inference_3pc.parties.client.numba_methods import private_compare_numba, post_compare_numba, \
    mult_client_numba

# TODO: change everything from dummy_tensors to dummy_tensor_shape - there is no need to pass dummy_tensors
import numpy as np
from torch.nn import Module


class SecureConv2DClient(SecureModule):

    def __init__(self, W_shape, stride, dilation, padding, groups, **kwargs):
        super(SecureConv2DClient, self).__init__(**kwargs)

        self.W_shape = W_shape
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups
        self.conv2d_handler = conv2d_handler_factory.create(self.device)
        self.is_dummy = False

        self.num_split_in_channels = NUM_SPLIT_CONV_IN_CHANNEL
        self.num_split_out_channels = NUM_SPLIT_CONV_OUT_CHANNEL

        self.out_channels, self.in_channels = self.W_shape[:2]

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

                outs.append(self.conv2d_handler.conv2d(X_share_splits[i],
                                                       F,
                                                       E,
                                                       W_share_splits[j][i],
                                                       padding=self.padding,
                                                       stride=self.stride,
                                                       dilation=self.dilation,
                                                       groups=self.groups))

            outs_all.append(np.concatenate(outs, axis=1))
        outs_all = np.stack(outs_all).sum(axis=0)
        return outs_all

        # outs = []
        # for i in range(div):
        #     start = i * mid
        #     end = (i + 1) * mid if i < div else None
        #
        #     F_share_server = self.network_assets.receiver_01.get()
        #     outs.append(self.conv2d_handler.conv2d
        #                 (X_share,
        #                  F_share_server + F_share[start:end],
        #                  E,
        #                  W_share[start:end],
        #                  padding=self.padding,
        #                  stride=self.stride,
        #                  dilation=self.dilation,
        #                  groups=self.groups))
        # out = np.concatenate(outs, axis=1)
        #
        #
        #
        # div = self.num_split_activation
        #
        # mid = W_share.shape[0] // div
        # self.network_assets.sender_01.put(E_share)
        # E_share_server = self.network_assets.receiver_01.get()
        # E = backend.add(E_share_server, E_share, out=E_share)
        #
        # for i in range(div):
        #     start = i * mid
        #     end = (i + 1) * mid if i < div else None
        #
        #     self.network_assets.sender_01.put(F_share[start:end])
        #
        # outs = []
        # for i in range(div):
        #     start = i * mid
        #     end = (i + 1) * mid if i < div else None
        #
        #     F_share_server = self.network_assets.receiver_01.get()
        #     outs.append(self.conv2d_handler.conv2d
        #                 (X_share,
        #                  F_share_server + F_share[start:end],
        #                  E,
        #                  W_share[start:end],
        #                  padding=self.padding,
        #                  stride=self.stride,
        #                  dilation=self.dilation,
        #                  groups=self.groups))
        # out = np.concatenate(outs, axis=1)
        # return out

    # @timer(name='client_conv2d')
    def forward(self, X_share):
        if self.is_dummy:
            out_shape = get_output_shape(X_share.shape, self.W_shape, self.padding, self.dilation, self.stride)
            self.network_assets.sender_01.put(X_share)
            mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=out_shape, dtype=SIGNED_DTYPE)
            return mu_0

        assert self.W_shape[2] == self.W_shape[3]
        assert (self.W_shape[1] == X_share.shape[1]) or self.groups > 1

        W_share = self.prf_handler[CLIENT, SERVER].integers(low=MIN_VAL, high=MAX_VAL, size=self.W_shape,
                                                            dtype=SIGNED_DTYPE)
        A_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=X_share.shape,
                                                                     dtype=SIGNED_DTYPE)
        B_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=W_share.shape,
                                                                     dtype=SIGNED_DTYPE)

        E_share = backend.subtract(X_share, A_share, out=A_share)
        F_share = backend.subtract(W_share, B_share, out=B_share)

        if self.num_split_out_channels == 1 and self.num_split_in_channels == 1:
            self.network_assets.sender_01.put(E_share)
            self.network_assets.sender_01.put(F_share)

            E_share_server = self.network_assets.receiver_01.get()
            F_share_server = self.network_assets.receiver_01.get()

            E = backend.add(E_share_server, E_share, out=E_share)
            F = backend.add(F_share_server, F_share, out=F_share)

            out = self.conv2d_handler.conv2d(X_share,
                                             F,
                                             E,
                                             W_share,
                                             padding=self.padding,
                                             stride=self.stride,
                                             dilation=self.dilation,
                                             groups=self.groups)
        else:
            out = self.split_conv(E_share, F_share, W_share, X_share)

        C_share = self.network_assets.receiver_02.get()
        out = backend.add(out, C_share, out=out)
        out = backend.right_shift(out, TRUNC_BITS, out=out)

        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=out.shape, dtype=SIGNED_DTYPE)

        out = backend.add(out, mu_0, out=out)

        return out


class PrivateCompareClient(SecureModule):
    def __init__(self, **kwargs):
        super(PrivateCompareClient, self).__init__(**kwargs)
        # self.decompose = Decompose(ignore_msb_bits=IGNORE_MSB_BITS, num_of_compare_bits=64,
        #                            dtype=SIGNED_DTYPE, **kwargs)

    def forward(self, x_bits_0, r, beta):
        s = self.prf_handler[CLIENT, SERVER].integers(low=1, high=P, size=x_bits_0.shape, dtype=backend.int8)
        d_bits_0 = private_compare_numba(s, r, x_bits_0, beta, IGNORE_MSB_BITS)
        # r[backend.astype(beta, backend.bool)] += 1
        # bits = self.decompose(r)
        # c_bits_0 = get_c_party_0(x_bits_0, bits, beta)
        # s = backend.multiply(s, c_bits_0, out=s)
        # d_bits_0 = module_67(s)

        d_bits_0 = self.prf_handler[CLIENT, SERVER].permutation(d_bits_0, axis=-1)

        self.network_assets.sender_02.put(d_bits_0)

        return


class ShareConvertClient(SecureModule):
    def __init__(self, **kwargs):
        super(ShareConvertClient, self).__init__(**kwargs)
        self.private_compare = PrivateCompareClient(**kwargs)

    def post_compare(self, a_0, eta_pp, delta_0, alpha, beta_0, mu_0, eta_p_0):
        return post_compare_numba(a_0, eta_pp, delta_0, alpha, beta_0, mu_0, eta_p_0)
        # eta_pp = backend.astype(eta_pp, SIGNED_DTYPE)
        # t0 = eta_pp * eta_p_0
        # t1 = self.add_mode_L_minus_one(t0, t0)
        # t2 = self.sub_mode_L_minus_one(eta_pp, t1)
        # eta_0 = self.add_mode_L_minus_one(eta_p_0, t2)
        #
        # t0 = self.add_mode_L_minus_one(delta_0, eta_0)
        # t1 = self.sub_mode_L_minus_one(t0, backend.ones_like(t0))
        # t2 = self.sub_mode_L_minus_one(t1, alpha)
        # theta_0 = self.add_mode_L_minus_one(beta_0, t2)
        #
        # y_0 = self.sub_mode_L_minus_one(a_0, theta_0)
        # y_0 = self.add_mode_L_minus_one(y_0, mu_0)
        #
        # return y_0

    def forward(self, a_0):
        eta_pp = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_0.shape, dtype=backend.int8)
        r = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=a_0.shape, dtype=SIGNED_DTYPE)
        r_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=a_0.shape, dtype=SIGNED_DTYPE)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=a_0.shape, dtype=SIGNED_DTYPE)
        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=list(a_0.shape) + [
            64 - IGNORE_MSB_BITS], dtype=backend.int8)

        alpha = backend.astype(0 < r_0 - r, SIGNED_DTYPE)
        a_tild_0 = a_0 + r_0
        self.network_assets.sender_02.put(a_tild_0)

        beta_0 = backend.astype(0 < a_0 - a_tild_0, SIGNED_DTYPE)
        delta_0 = self.network_assets.receiver_02.get()

        self.private_compare(x_bits_0, r - 1, eta_pp)
        eta_p_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=a_0.shape,
                                                                     dtype=SIGNED_DTYPE)

        return self.post_compare(a_0, eta_pp, delta_0, alpha, beta_0, mu_0, eta_p_0)


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


class SecurePostBReLUMultClient(SecureModule):
    def __init__(self, **kwargs):
        super(SecurePostBReLUMultClient, self).__init__(**kwargs)

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

        A_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1,
                                                                     size=non_identity_activation.shape,
                                                                     dtype=SIGNED_DTYPE)
        B_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=sign_tensors.shape,
                                                                     dtype=SIGNED_DTYPE)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=activation.shape, dtype=SIGNED_DTYPE)

        E_share = non_identity_activation - A_share
        F_share = sign_tensors - B_share

        self.network_assets.sender_01.put(E_share)
        E_share_server = self.network_assets.receiver_01.get()

        self.network_assets.sender_01.put(F_share)
        F_share_server = self.network_assets.receiver_01.get()

        C_share = self.network_assets.receiver_02.get()
        E = E_share_server + E_share
        F = F_share_server + F_share

        F = self.post(activation, F, cumsum_shapes, pad_handlers, active_block_sizes, active_block_sizes_to_channels)[:,
            ~is_identity_channels]
        sign_tensors = self.post(activation, sign_tensors, cumsum_shapes, pad_handlers, active_block_sizes,
                                 active_block_sizes_to_channels)[:, ~is_identity_channels]

        out = non_identity_activation * F + sign_tensors * E + C_share
        activation[:, ~is_identity_channels] = out

        activation = activation + mu_0

        return activation


class SecureMultiplicationClient(SecureModule):
    def __init__(self, **kwargs):
        super(SecureMultiplicationClient, self).__init__(**kwargs)

    def exchange_shares(self, E_share, F_share):
        self.network_assets.sender_01.put(E_share)
        E_share_server = self.network_assets.receiver_01.get()

        self.network_assets.sender_01.put(F_share)
        F_share_server = self.network_assets.receiver_01.get()

        return E_share_server, F_share_server

    def forward(self, X_share, Y_share):
        A_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape,
                                                                     dtype=SIGNED_DTYPE)
        B_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape,
                                                                     dtype=SIGNED_DTYPE)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=X_share.shape, dtype=SIGNED_DTYPE)

        E_share = X_share - A_share
        F_share = Y_share - B_share

        E_share_server, F_share_server = self.exchange_shares(E_share, F_share)

        C_share = self.network_assets.receiver_02.get()
        out = mult_client_numba(X_share, Y_share, C_share, mu_0, E_share, E_share_server, F_share, F_share_server)
        # E = E_share_server + E_share
        # F = F_share_server + F_share
        #
        # out = X_share * F + Y_share * E + C_share
        # out = out + mu_0
        return out


class SecureSelectShareClient(SecureModule):
    def __init__(self, **kwargs):
        super(SecureSelectShareClient, self).__init__(**kwargs)
        self.secure_multiplication = SecureMultiplicationClient(**kwargs)

    def forward(self, alpha, x, y):
        # if alpha == 0: return x else return 1
        shape = alpha.shape
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)

        w = y - x
        c = self.secure_multiplication(alpha, w)
        z = x + c
        return z + mu_0


class SecureMSBClient(SecureModule):
    def __init__(self, **kwargs):
        super(SecureMSBClient, self).__init__(**kwargs)
        self.mult = SecureMultiplicationClient(**kwargs)
        self.private_compare = PrivateCompareClient(**kwargs)

    def post_compare(self, beta, x_bit_0_0, r_mode_2, mu_0):
        beta = backend.astype(beta, SIGNED_DTYPE)
        beta_p_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=mu_0.shape,
                                                                      dtype=SIGNED_DTYPE)

        gamma_0 = beta_p_0 - (2 * beta * beta_p_0)
        delta_0 = x_bit_0_0 - (2 * r_mode_2 * x_bit_0_0)

        theta_0 = self.mult(gamma_0, delta_0)

        alpha_0 = gamma_0 + delta_0 - 2 * theta_0
        alpha_0 = alpha_0 + mu_0

        return alpha_0

    def pre_compare(self, a_0):
        beta = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_0.shape, dtype=backend.int8)
        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=list(a_0.shape) + [
            64 - IGNORE_MSB_BITS], dtype=backend.int8)
        x_bit_0_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=a_0.shape,
                                                                       dtype=SIGNED_DTYPE)

        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=a_0.shape, dtype=a_0.dtype)

        x_0 = self.network_assets.receiver_02.get()
        r_1 = self.network_assets.receiver_01.get()

        y_0 = self.add_mode_L_minus_one(a_0, a_0)
        r_0 = self.add_mode_L_minus_one(x_0, y_0)
        self.network_assets.sender_01.put(r_0)
        r = self.add_mode_L_minus_one(r_0, r_1)

        r_mode_2 = r % 2

        return x_bits_0, r, beta, x_bit_0_0, r_mode_2, mu_0

    def forward(self, a_0):
        x_bits_0, r, beta, x_bit_0_0, r_mode_2, mu_0 = self.pre_compare(a_0)

        self.private_compare(x_bits_0, r, beta)

        return self.post_compare(beta, x_bit_0_0, r_mode_2, mu_0)


class SecureDReLUClient(SecureModule):
    # counter = 0
    def __init__(self, **kwargs):
        super(SecureDReLUClient, self).__init__(**kwargs)

        self.share_convert = ShareConvertClient(**kwargs)
        self.msb = SecureMSBClient(**kwargs)

    def forward(self, X_share):
        # SecureDReLUClient.counter += 1
        # np.save("/home/yakir/Data2/secure_activation_statistics/client/{}.npy".format(SecureDReLUClient.counter), X_share)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape, dtype=SIGNED_DTYPE)

        X0_converted = self.share_convert(X_share)
        MSB_0 = self.msb(X0_converted)

        return -MSB_0 + mu_0


class SecureReLUClient(SecureModule):
    def __init__(self, dummy_relu=False, **kwargs):
        super(SecureReLUClient, self).__init__(**kwargs)

        self.DReLU = SecureDReLUClient(**kwargs)
        self.mult = SecureMultiplicationClient(**kwargs)
        self.dummy_relu = dummy_relu

    def forward(self, X_share):
        # return X_share
        if self.dummy_relu:
            self.network_assets.sender_01.put(X_share)
            mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape,
                                                             dtype=SIGNED_DTYPE)
            return mu_0
        else:

            shape = X_share.shape
            mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)

            X_share = X_share.flatten()
            MSB_0 = self.DReLU(X_share)
            ret = self.mult(X_share, MSB_0).reshape(shape)

            return ret + mu_0


class SecureMaxPoolClient(SecureMaxPool):
    def __init__(self, kernel_size=3, stride=2, padding=1, **kwargs):
        super(SecureMaxPoolClient, self).__init__(kernel_size, stride, padding, **kwargs)
        self.select_share = SecureSelectShareClient(**kwargs)
        self.dReLU = SecureDReLUClient(**kwargs)
        self.mult = SecureMultiplicationClient(**kwargs)

    def forward(self, x):
        ret = super(SecureMaxPoolClient, self).forward(x)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=ret.shape, dtype=SIGNED_DTYPE)

        return ret + mu_0


class SecureBlockReLUClient(SecureModule, SecureOptimizedBlockReLU):
    def __init__(self, block_sizes, dummy_relu=False, **kwargs):
        SecureModule.__init__(self, **kwargs)
        SecureOptimizedBlockReLU.__init__(self, block_sizes)
        self.DReLU = SecureDReLUClient(**kwargs)
        self.mult = SecureMultiplicationClient(**kwargs)
        self.post_bReLU = SecurePostBReLUMultClient(**kwargs)
        self.dummy_relu = dummy_relu

    def forward(self, activation):
        if self.dummy_relu:
            return activation
        return SecureOptimizedBlockReLU.forward(self, activation)