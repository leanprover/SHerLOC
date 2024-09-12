"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3xf16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3xf16>
    %4 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f16>}> : () -> tensor<f16>
    %5 = "stablehlo.reduce"(%2, %4) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %6 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      "stablehlo.return"(%6) : (tensor<f16>) -> ()
    }) : (tensor<2x3xf16>, tensor<f16>) -> tensor<3xf16>
    "stablehlo.custom_call"(%5, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3xf16>, tensor<3xf16>) -> ()
    "func.return"(%5) : (tensor<3xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-1.045900e+00, -2.212890e+00, 9.575190e-01], [-2.121090e+00, 1.873050e+00, 4.433590e+00]]> : tensor<2x3xf16>}> : () -> tensor<2x3xf16>
    "func.return"(%1) : (tensor<2x3xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[2.218750e+00, -4.144530e+00, 4.246090e+00]> : tensor<3xf16>}> : () -> tensor<3xf16>
    "func.return"(%0) : (tensor<3xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

