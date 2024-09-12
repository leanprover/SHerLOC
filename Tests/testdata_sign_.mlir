"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x2xui16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<2x2xui16>}> : () -> tensor<2x2xui16>
    %2 = "func.call"() <{callee = @expected}> : () -> tensor<2x2xui16>
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<ui16>}> : () -> tensor<ui16>
    %4 = "stablehlo.broadcast_in_dim"(%3) <{broadcast_dimensions = array<i64>}> : (tensor<ui16>) -> tensor<2x2xui16>
    %5 = "stablehlo.compare"(%1, %4) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<2x2xui16>, tensor<2x2xui16>) -> tensor<2x2xi1>
    %6 = "stablehlo.constant"() <{value = dense<0> : tensor<ui16>}> : () -> tensor<ui16>
    %7 = "stablehlo.broadcast_in_dim"(%6) <{broadcast_dimensions = array<i64>}> : (tensor<ui16>) -> tensor<2x2xui16>
    %8 = "stablehlo.constant"() <{value = dense<1> : tensor<ui16>}> : () -> tensor<ui16>
    %9 = "stablehlo.broadcast_in_dim"(%8) <{broadcast_dimensions = array<i64>}> : (tensor<ui16>) -> tensor<2x2xui16>
    %10 = "stablehlo.select"(%5, %7, %9) : (tensor<2x2xi1>, tensor<2x2xui16>, tensor<2x2xui16>) -> tensor<2x2xui16>
    "stablehlo.custom_call"(%10, %2) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<2x2xui16>, tensor<2x2xui16>) -> ()
    "func.return"(%10) : (tensor<2x2xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x2xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<0> : tensor<2x2xui16>}> : () -> tensor<2x2xui16>
    "func.return"(%0) : (tensor<2x2xui16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

