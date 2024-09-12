"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x4x6xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x3x5xf16>, tensor<2x4x6xf16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2x4x6xf16>
    %5 = "stablehlo.constant"() <{value = dense<0xFC00> : tensor<f16>}> : () -> tensor<f16>
    %6 = "stablehlo.pad"(%3#1, %5) <{edge_padding_high = array<i64: 0, 0, 0>, edge_padding_low = array<i64: 0, 0, 0>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<2x4x6xf16>, tensor<f16>) -> tensor<2x4x6xf16>
    %7 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f16>}> : () -> tensor<f16>
    %8 = "stablehlo.select_and_scatter"(%6, %3#0, %7) <{window_dimensions = array<i64: 2, 2, 2>}> ({
    ^bb0(%arg2: tensor<f16>, %arg3: tensor<f16>):
      %11 = "stablehlo.compare"(%arg2, %arg3) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<f16>, tensor<f16>) -> tensor<i1>
      "stablehlo.return"(%11) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %10 = "stablehlo.add"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      "stablehlo.return"(%10) : (tensor<f16>) -> ()
    }) : (tensor<2x4x6xf16>, tensor<1x3x5xf16>, tensor<f16>) -> tensor<2x4x6xf16>
    %9 = "stablehlo.slice"(%8) <{limit_indices = array<i64: 2, 4, 6>, start_indices = array<i64: 0, 0, 0>, strides = array<i64: 1, 1, 1>}> : (tensor<2x4x6xf16>) -> tensor<2x4x6xf16>
    "stablehlo.custom_call"(%9, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x4x6xf16>, tensor<2x4x6xf16>) -> ()
    "func.return"(%9) : (tensor<2x4x6xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x3x5xf16>, tensor<2x4x6xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[-3.878910e+00, 1.413090e+00, 4.476560e+00, -3.933590e+00, 6.054690e-02], [-7.019530e+00, -1.442380e+00, 4.257810e+00, -3.005860e+00, 2.126950e+00], [4.484380e+00, -3.417970e-02, -1.207030e+00, 4.562500e+00, 3.677730e+00]]]> : tensor<1x3x5xf16>}> : () -> tensor<1x3x5xf16>
    %2 = "stablehlo.constant"() <{value = dense<[[[-1.462890e+00, -3.843750e+00, 2.527340e+00, -5.609380e+00, -3.707030e+00, -2.681640e+00], [-1.667970e+00, 1.726560e+00, 3.255860e+00, 5.164060e+00, -1.097660e+00, 1.143190e-01], [-2.068360e+00, -1.214840e+00, -4.921880e+00, 1.913090e+00, -3.544920e+00, 3.684080e-01], [-1.171880e-01, -4.000000e+00, -1.446290e+00, -5.804690e+00, -6.160150e+00, -4.410160e+00]], [[-2.937500e+00, 1.557920e-02, 7.915030e-01, 5.996090e+00, 4.687500e+00, -3.720700e+00], [3.645020e-01, -6.455080e-01, 3.906250e+00, 5.367190e+00, 2.393800e-01, 1.419680e-01], [1.138670e+00, 2.181640e+00, -2.730470e+00, 5.585940e-01, 1.885740e+00, -3.236330e+00], [-4.449220e+00, -1.608400e+00, 9.716790e-01, -6.566400e+00, 2.250000e+00, 3.755860e+00]]]> : tensor<2x4x6xf16>}> : () -> tensor<2x4x6xf16>
    "func.return"(%1, %2) : (tensor<1x3x5xf16>, tensor<2x4x6xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x4x6xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, -3.878910e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, -1.207030e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00, 0.000000e+00, 5.429690e-01, 6.054690e-02, 0.000000e+00], [0.000000e+00, 0.000000e+00, -2.929690e-02, 1.251950e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, -2.570310e+00, 0.000000e+00, 0.000000e+00, 2.126950e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 4.562500e+00, 3.677730e+00]]]> : tensor<2x4x6xf16>}> : () -> tensor<2x4x6xf16>
    "func.return"(%0) : (tensor<2x4x6xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

