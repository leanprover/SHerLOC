"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xi1>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<f32>, tensor<2x3xf32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xi1>
    %5 = "stablehlo.broadcast_in_dim"(%3#0) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<2x3xf32>
    %6 = "stablehlo.compare"(%5, %3#1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xi1>
    "stablehlo.custom_call"(%6, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<2x3xi1>, tensor<2x3xi1>) -> ()
    "func.return"(%6) : (tensor<2x3xi1>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<f32>, tensor<2x3xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0.0436727554, -2.03590417, 1.36341751], [-0.444336385, 2.11136985, -2.04138827]]> : tensor<2x3xf32>}> : () -> tensor<2x3xf32>
    %2 = "stablehlo.constant"() <{value = dense<-2.5417707> : tensor<f32>}> : () -> tensor<f32>
    "func.return"(%2, %1) : (tensor<f32>, tensor<2x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xi1>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<false> : tensor<2x3xi1>}> : () -> tensor<2x3xi1>
    "func.return"(%0) : (tensor<2x3xi1>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

