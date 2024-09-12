"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x33x2xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x16x2xf32>, tensor<3x2x2xf32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<1x33x2xf32>
    %5 = "stablehlo.convolution"(%3#0, %3#1) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, f]x[0, i, o]->[b, 0, f]>, feature_group_count = 1 : i64, lhs_dilation = array<i64: 2>, padding = dense<2> : tensor<1x2xi64>}> : (tensor<1x16x2xf32>, tensor<3x2x2xf32>) -> tensor<1x33x2xf32>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_almost_eq", has_side_effect = true}> : (tensor<1x33x2xf32>, tensor<1x33x2xf32>) -> ()
    "func.return"(%5) : (tensor<1x33x2xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x16x2xf32>, tensor<3x2x2xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[-2.24425125, -0.326069951], [1.11427498, 4.61250925], [4.420030e+00, 0.840528547], [1.07607019, -3.43172455], [-1.61881506, 3.50739169], [2.90553093, 0.567650259], [-8.65767383, -0.601767719], [-1.36898816, 1.11275971], [-2.23143911, 0.242084026], [-6.81402922, -4.01505184], [1.24825752, 2.79364014], [-2.95937228, -2.14742327], [-1.26819241, -1.11193192], [2.64123178, -1.92954421], [-2.75114107, 1.74770582], [-5.94057512, -1.25173676]]]> : tensor<1x16x2xf32>}> : () -> tensor<1x16x2xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[[-1.16260839, -1.83661807], [-0.214550629, 0.93904066]], [[-2.58689475, 7.01376152], [2.17989922, 3.44552255]], [[-0.495644063, 0.346568316], [-1.58046222, 1.97559404]]]> : tensor<3x2x2xf32>}> : () -> tensor<3x2x2xf32>
    "func.return"(%1, %2) : (tensor<1x16x2xf32>, tensor<3x2x2xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x33x2xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[1.62769103, -1.42196822], [5.09484196, -16.8641243], [-5.16303682, 13.3142567], [7.17229366, 23.7077637], [-5.80426741, 5.47722149], [-9.60188484, 3.389710e+01], [-0.42873621, -13.7353802], [-10.2644939, -4.27678442], [-5.25571394, 1.16929317], [11.8334646, 7.308140e-01], [-1.20772398, 8.39513969], [-6.27888202, 22.3345528], [1.74241161, -8.99262809], [21.0846977, -62.7962646], [9.1144495, 17.0596695], [5.96713257, -5.76771736], [2.07624865, 3.26414871], [6.30021667, -14.8166752], [12.2653217, -5.9680109], [8.87476825, -61.6259307], [3.74954534, 14.6961784], [2.86074305, 18.3805313], [2.81010461, -4.9372921], [2.97441864, -28.1553268], [6.287261, 0.782477378], [0.856780767, -12.7259855], [3.45343781, -1.61159301], [-11.0388012, 11.8766804], [-4.055330e+00, -4.16355562], [10.9267349, -13.274087], [7.74626159, 2.16222358], [12.6389828, -45.9786606], [7.17512321, 9.73513603]]]> : tensor<1x33x2xf32>}> : () -> tensor<1x33x2xf32>
    "func.return"(%0) : (tensor<1x33x2xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

