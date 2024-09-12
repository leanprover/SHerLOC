"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<6x4xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xf16>, tensor<f16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<6x4xf16>
    %5 = "stablehlo.pad"(%3#0, %3#1) <{edge_padding_high = array<i64: 2, 1>, edge_padding_low = array<i64: 1, 0>, interior_padding = array<i64: 1, 0>}> : (tensor<2x3xf16>, tensor<f16>) -> tensor<6x4xf16>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<6x4xf16>, tensor<6x4xf16>) -> ()
    "func.return"(%5) : (tensor<6x4xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xf16>, tensor<f16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[1.447680e-03, 1.801250e-04, -2.915860e-04], [6.337160e-04, 6.279940e-04, 7.696150e-04]]> : tensor<2x3xf16>}> : () -> tensor<2x3xf16>
    %2 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f16>}> : () -> tensor<f16>
    "func.return"(%1, %2) : (tensor<2x3xf16>, tensor<f16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<6x4xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [1.447680e-03, 1.801250e-04, -2.915860e-04, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [6.337160e-04, 6.279940e-04, 7.696150e-04, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<6x4xf16>}> : () -> tensor<6x4xf16>
    "func.return"(%0) : (tensor<6x4xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

