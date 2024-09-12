"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %5:4 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xi32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>)
    %6 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xf32>
    %7 = "stablehlo.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    %8 = "stablehlo.broadcast_in_dim"(%7) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<2x3xi32>
    %9 = "stablehlo.compare"(%5#0, %8) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %10 = "stablehlo.constant"() <{value = dense<2> : tensor<i32>}> : () -> tensor<i32>
    %11 = "stablehlo.broadcast_in_dim"(%10) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<2x3xi32>
    %12 = "stablehlo.compare"(%5#0, %11) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %13 = "stablehlo.select"(%12, %5#2, %5#3) : (tensor<2x3xi1>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %14 = "stablehlo.select"(%9, %5#1, %13) : (tensor<2x3xi1>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    "stablehlo.custom_call"(%14, %6) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x3xf32>, tensor<2x3xf32>) -> ()
    "func.return"(%14) : (tensor<2x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xi32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0, 0, 2], [2, 1, 0]]> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
    %2 = "stablehlo.constant"() <{value = dense<[[3.41892362, -0.968371093, -3.74343753], [-1.74075079, -6.24598694, 1.09240973]]> : tensor<2x3xf32>}> : () -> tensor<2x3xf32>
    %3 = "stablehlo.constant"() <{value = dense<[[5.66154289, -2.86463428, 0.038875252], [1.28850126, 0.555465043, 4.92511129]]> : tensor<2x3xf32>}> : () -> tensor<2x3xf32>
    %4 = "stablehlo.constant"() <{value = dense<[[2.02817249, -5.940207, -2.58250499], [3.40481019, 5.68319035, 0.296268404]]> : tensor<2x3xf32>}> : () -> tensor<2x3xf32>
    "func.return"(%1, %2, %3, %4) : (tensor<2x3xi32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[3.41892362, -0.968371093, -2.58250499], [3.40481019, 0.555465043, 1.09240973]]> : tensor<2x3xf32>}> : () -> tensor<2x3xf32>
    "func.return"(%0) : (tensor<2x3xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

