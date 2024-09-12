"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<18xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %5:4 = "func.call"() <{callee = @inputs}> : () -> (tensor<18xi32>, tensor<18xf32>, tensor<18xf32>, tensor<18xf32>)
    %6 = "func.call"() <{callee = @expected}> : () -> tensor<18xf32>
    %7 = "stablehlo.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    %8 = "stablehlo.broadcast_in_dim"(%7) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<18xi32>
    %9 = "stablehlo.compare"(%5#0, %8) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<18xi32>, tensor<18xi32>) -> tensor<18xi1>
    %10 = "stablehlo.constant"() <{value = dense<2> : tensor<i32>}> : () -> tensor<i32>
    %11 = "stablehlo.broadcast_in_dim"(%10) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<18xi32>
    %12 = "stablehlo.compare"(%5#0, %11) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<18xi32>, tensor<18xi32>) -> tensor<18xi1>
    %13 = "stablehlo.select"(%12, %5#2, %5#3) : (tensor<18xi1>, tensor<18xf32>, tensor<18xf32>) -> tensor<18xf32>
    %14 = "stablehlo.select"(%9, %5#1, %13) : (tensor<18xi1>, tensor<18xf32>, tensor<18xf32>) -> tensor<18xf32>
    "stablehlo.custom_call"(%14, %6) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<18xf32>, tensor<18xf32>) -> ()
    "func.return"(%14) : (tensor<18xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<18xi32>, tensor<18xf32>, tensor<18xf32>, tensor<18xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[0, 2, 0, 0, 1, 1, 2, 0, 1, 0, 0, 1, 2, 1, 2, 1, 1, 0]> : tensor<18xi32>}> : () -> tensor<18xi32>
    %2 = "stablehlo.constant"() <{value = dense<[2.56200647, 0.303029031, -2.33659768, -0.0842525735, -3.8680172, 2.72818685, -1.30297685, -4.97471142, -2.56493282, -5.70271301, -1.74544048, 3.00412154, -0.551256299, 3.3540206, 1.73112082, 2.13606191, -2.09324622, -0.971786737]> : tensor<18xf32>}> : () -> tensor<18xf32>
    %3 = "stablehlo.constant"() <{value = dense<[-6.02375221, 2.72696185, 3.9226594, 1.65064454, -2.43719792, 0.226952314, 0.855229318, -1.46683788, 0.0280455742, -1.52327204, -0.0184807125, 0.266047239, -0.881479442, -1.93082952, 4.86475706, -1.67108071, -2.33280182, 5.68302345]> : tensor<18xf32>}> : () -> tensor<18xf32>
    %4 = "stablehlo.constant"() <{value = dense<[-0.763047457, 3.95658016, 6.675350e-01, 0.90997231, -5.72888422, -1.85249972, -0.0301251821, -0.243915722, 0.0369620621, -0.119632974, -1.40885472, -3.06102085, -1.34765267, -2.44184375, 0.426090688, 0.480808973, 0.913791298, -0.0524138957]> : tensor<18xf32>}> : () -> tensor<18xf32>
    "func.return"(%1, %2, %3, %4) : (tensor<18xi32>, tensor<18xf32>, tensor<18xf32>, tensor<18xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<18xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[2.56200647, 3.95658016, -2.33659768, -0.0842525735, -2.43719792, 0.226952314, -0.0301251821, -4.97471142, 0.0280455742, -5.70271301, -1.74544048, 0.266047239, -1.34765267, -1.93082952, 0.426090688, -1.67108071, -2.33280182, -0.971786737]> : tensor<18xf32>}> : () -> tensor<18xf32>
    "func.return"(%0) : (tensor<18xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

