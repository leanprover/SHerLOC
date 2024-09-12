"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xf16>, tensor<5x2x2xf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      "stablehlo.return"(%7) : (tensor<f16>) -> ()
    }) : (tensor<5x6x7xf16>, tensor<2x2x2xi64>, tensor<5x2x2xf16>) -> tensor<5x6x7xf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<5x6x7xf16>, tensor<5x6x7xf16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xf16>, tensor<5x2x2xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x9F32D7C300C28B2EFC428AC5744515BA6FBF5B232AC4E1BEF1BA63B0D240F3461E3AE53D60BD46C2BEC69F40813B3C45ADC20E48E536143E18BF5B36113C44B76B3EC9BE02384B4811C53AC2FCC0B1C1E9B26DC065C4DC445341D8427F42FAA86EBE0A449FC09834733A5EB508B9E5BDE7430F3D473DF1B898BBFEB8A9B8E93D1AC6A1C0A24329A6323A3DC01C41A8329C39FAB856C47CC5353A08B242BC8CB5424513C07CC386364C1ED9C210C383445642C4359ABF1A40F82A014139BC9C3EFF3A9939CA3A8EC11F44BB4394B18CC3B6C0EEC1C84221B63C38FA40A0421E36A5C029C68AB5C2C33B42B8415A422FC4B2406DBB40421CC09840BDC4FA4586BCBE402E3D7B45C6BDBAAC3BB4F6C7F2B4B04260AC4940FB3CC4387FB5C2C521BFB03602C5393F98BF353BBCC1CDC0853F8A400DB54FC5D4444A4282443F3BAE3886B5544454399A3B51C55F42F0C465C062C0C13BEA2CE3B01F2EA940E3BE04BB6FC369C1083FD438EEC4E2C0D0AFF0A52E446FC05EBC3DBCECB0FDBCE84209C1D938C73D2A3FCFBC29C48B3CC2428940E8C0253CAE476BB61B29544216411AC127BFBEB5"> : tensor<5x6x7xf16>}> : () -> tensor<5x6x7xf16>
    %2 = "stablehlo.constant"() <{value = dense<[[[2.064450e+00, -1.041020e+00], [2.996090e+00, 6.762700e-01]], [[-3.753910e+00, -6.384270e-02], [-2.867190e+00, 3.369140e+00]], [[4.679690e+00, -1.494140e+00], [2.324220e+00, -1.407470e-01]], [[-1.486330e+00, 3.533200e+00], [9.033200e-01, -1.163940e-01]], [[-2.361330e+00, 1.148680e-01], [-1.842770e+00, -2.687500e+00]]]> : tensor<5x2x2xf16>}> : () -> tensor<5x2x2xf16>
    "func.return"(%1, %2) : (tensor<5x6x7xf16>, tensor<5x2x2xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x9F320CC800C28B2EFC428AC5744515BA6FBFF9202AC4E1BEF1BA63B0D240F3461E3A23BE60BD46C2BEC69F40813B3C45ADC20E48E536143E50C55B36113C44B76B3EC9BE02384B4811C53AC2FCC0B1C1E9B26DC065C48FCC5341D8427F42FAA86EBE0A449FC0BD3B733A5EB508B9E5BDE7430F3D473D0C2998BBFEB8A9B8E93D1AC6A1C0A24329A6323A3DC053C7A8329C39FAB856C47CC5353A08B242BC8CB5424513C07CC386364C1E01CC10C383445642C4359ABF1A40F82AA2B539BC9C3EFF3A9939CA3A8EC11F44C6C594B18CC3B6C0EEC1C84221B63C38FA40A0421E3666C529C68AB5C2C33B42B8415A422FC4B2406DBB40421CC09840BDC4FA45B93EBE402E3D7B45C6BDBAAC3BB4F6C79B28B04260AC4940FB3CC4387FB5C2C54CC6B03602C5393F98BF353BBCC1CDC0853F8A400DB5CCC4D4444A4282443F3BAE3886B5544454399A3B51C55F42F0C465C062C094C0EA2CE3B01F2EA940E3BE04BB6FC34547083FD438EEC4E2C0D0AFF0A52E4413B45EBC3DBCECB0FDBCE84209C1D938C73D2A3FCFBCAB478B3CC2428940E8C0253CAE476BB61B29544216411AC127BFBEB5"> : tensor<5x6x7xf16>}> : () -> tensor<5x6x7xf16>
    "func.return"(%0) : (tensor<5x6x7xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

