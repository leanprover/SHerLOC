"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3xui16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3xui16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3xui16>
    %4 = "stablehlo.constant"() <{value = dense<1> : tensor<ui16>}> : () -> tensor<ui16>
    %5 = "stablehlo.reduce"(%2, %4) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %6 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<ui16>, tensor<ui16>) -> tensor<ui16>
      "stablehlo.return"(%6) : (tensor<ui16>) -> ()
    }) : (tensor<2x3xui16>, tensor<ui16>) -> tensor<3xui16>
    "stablehlo.custom_call"(%5, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<3xui16>, tensor<3xui16>) -> ()
    "func.return"(%5) : (tensor<3xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[3, 3, 2], [3, 0, 0]]> : tensor<2x3xui16>}> : () -> tensor<2x3xui16>
    "func.return"(%1) : (tensor<2x3xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[9, 0, 0]> : tensor<3xui16>}> : () -> tensor<3xui16>
    "func.return"(%0) : (tensor<3xui16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

