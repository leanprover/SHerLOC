"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<8x9xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %6 = "func.call"() <{callee = @inputs}> : () -> tensor<8x9xf32>
    %7 = "func.call"() <{callee = @expected}> : () -> tensor<8x9xf32>
    %8 = "func.call"(%6) <{callee = @cummax}> : (tensor<8x9xf32>) -> tensor<8x9xf32>
    "stablehlo.custom_call"(%8, %7) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<8x9xf32>, tensor<8x9xf32>) -> ()
    "func.return"(%8) : (tensor<8x9xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %5 = "stablehlo.constant"() <{value = dense<[[1.46896458, -1.83834624, 1.91261089, -5.147048, -0.333510429, 6.64224195, 0.353656948, 2.31035757, -1.77273178], [4.01471615, 2.34389973, -1.06755078, -4.94970608, 0.861732542, -1.94748127, 1.12188435, -3.7747016, 2.52044225], [3.19555068, 1.45463204, 0.5395841, -0.384640962, -1.01408541, 1.848430e+00, -2.12792468, -2.1028471, -5.81991196], [-4.1881547, 3.86492443, 0.598357797, -1.56349504, -2.21620631, -4.60625648, 4.5639925, 3.33661318, -2.6640923], [-3.02472973, -0.601977229, 5.57161236, 0.451104701, -2.8828764, 0.263059527, 0.350622714, 3.79335141, -4.49060917], [-0.682265579, 0.112245716, -1.92434478, 4.74856234, -3.45426893, 0.811877131, -0.841498732, 6.753820e-02, 1.90612876], [1.185583, 0.17747426, 0.163559556, 2.29865241, 4.10513878, 1.05198336, -1.94492424, 3.36212707, -3.72425747], [1.303040e+00, 6.04134321, 0.34423548, 3.42879128, -3.69795394, 0.233644933, 3.33242822, -5.93633795, 0.223473743]]> : tensor<8x9xf32>}> : () -> tensor<8x9xf32>
    "func.return"(%5) : (tensor<8x9xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %4 = "stablehlo.constant"() <{value = dense<[[1.46896458, -1.83834624, 1.91261089, -5.147048, -0.333510429, 6.64224195, 0.353656948, 2.31035757, -1.77273178], [4.01471615, 2.34389973, 1.91261089, -4.94970608, 0.861732542, 6.64224195, 1.12188435, 2.31035757, 2.52044225], [4.01471615, 2.34389973, 1.91261089, -0.384640962, 0.861732542, 6.64224195, 1.12188435, 2.31035757, 2.52044225], [4.01471615, 3.86492443, 1.91261089, -0.384640962, 0.861732542, 6.64224195, 4.5639925, 3.33661318, 2.52044225], [4.01471615, 3.86492443, 5.57161236, 0.451104701, 0.861732542, 6.64224195, 4.5639925, 3.79335141, 2.52044225], [4.01471615, 3.86492443, 5.57161236, 4.74856234, 0.861732542, 6.64224195, 4.5639925, 3.79335141, 2.52044225], [4.01471615, 3.86492443, 5.57161236, 4.74856234, 4.10513878, 6.64224195, 4.5639925, 3.79335141, 2.52044225], [4.01471615, 6.04134321, 5.57161236, 4.74856234, 4.10513878, 6.64224195, 4.5639925, 3.79335141, 2.52044225]]> : tensor<8x9xf32>}> : () -> tensor<8x9xf32>
    "func.return"(%4) : (tensor<8x9xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x9xf32>) -> tensor<8x9xf32>, sym_name = "cummax", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<8x9xf32>):
    %0 = "stablehlo.constant"() <{value = dense<0xFF800000> : tensor<f32>}> : () -> tensor<f32>
    %1 = "stablehlo.broadcast_in_dim"(%0) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<f32>
    %2 = "stablehlo.reduce_window"(%arg0, %1) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %3 = "stablehlo.maximum"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%3) : (tensor<f32>) -> ()
    }) : (tensor<8x9xf32>, tensor<f32>) -> tensor<8x9xf32>
    "func.return"(%2) : (tensor<8x9xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

