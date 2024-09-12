"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3xbf16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3xbf16>
    %4 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
    %5 = "stablehlo.reduce"(%2, %4) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %6 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%6) : (tensor<bf16>) -> ()
    }) : (tensor<2x3xbf16>, tensor<bf16>) -> tensor<3xbf16>
    "stablehlo.custom_call"(%5, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3xbf16>, tensor<3xbf16>) -> ()
    "func.return"(%5) : (tensor<3xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-4.937500e+00, -2.412110e-01, -5.593750e+00], [-3.015630e+00, 9.187500e+00, -1.375000e+00]]> : tensor<2x3xbf16>}> : () -> tensor<2x3xbf16>
    "func.return"(%1) : (tensor<2x3xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[1.487500e+01, -2.218750e+00, 7.687500e+00]> : tensor<3xbf16>}> : () -> tensor<3xbf16>
    "func.return"(%0) : (tensor<3xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

