"builtin.module"() ({
  "builtin.module"() <{sym_name = "cross_replica"}> ({
    "func.func"() <{function_type = (tensor<2x4xi64>) -> tensor<4x2xi64>, sym_name = "all_to_all"}> ({
    ^bb0(%arg5: tensor<2x4xi64>):
      %21 = "stablehlo.all_to_all"(%arg5) <{concat_dimension = 0 : i64, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, split_count = 2 : i64, split_dimension = 1 : i64}> : (tensor<2x4xi64>) -> tensor<4x2xi64>
      "func.return"(%21) : (tensor<4x2xi64>) -> ()
    }) : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
      %18 = "stablehlo.constant"() <{value = dense<[[1, 2, 3, 4], [5, 6, 7, 8]]> : tensor<2x4xi64>}> : () -> tensor<2x4xi64>
      %19 = "stablehlo.constant"() <{value = dense<[[9, 10, 11, 12], [13, 14, 15, 16]]> : tensor<2x4xi64>}> : () -> tensor<2x4xi64>
      %20:2 = "interpreter.run_parallel"(%18, %19) <{programs = [[@all_to_all], [@all_to_all]]}> : (tensor<2x4xi64>, tensor<2x4xi64>) -> (tensor<4x2xi64>, tensor<4x2xi64>)
      "check.expect_eq_const"(%20#0) <{value = dense<[[1, 2], [5, 6], [9, 10], [13, 14]]> : tensor<4x2xi64>}> : (tensor<4x2xi64>) -> ()
      "check.expect_eq_const"(%20#1) <{value = dense<[[3, 4], [7, 8], [11, 12], [15, 16]]> : tensor<4x2xi64>}> : (tensor<4x2xi64>) -> ()
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() <{sym_name = "cross_replica_issue_2433"}> ({
    "func.func"() <{function_type = (tensor<2x4xi64>) -> tensor<4x2xi64>, sym_name = "all_to_all"}> ({
    ^bb0(%arg4: tensor<2x4xi64>):
      %17 = "stablehlo.all_to_all"(%arg4) <{concat_dimension = 0 : i64, replica_groups = dense<[[1, 0]]> : tensor<1x2xi64>, split_count = 2 : i64, split_dimension = 1 : i64}> : (tensor<2x4xi64>) -> tensor<4x2xi64>
      "func.return"(%17) : (tensor<4x2xi64>) -> ()
    }) : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
      %14 = "stablehlo.constant"() <{value = dense<[[1, 2, 3, 4], [5, 6, 7, 8]]> : tensor<2x4xi64>}> : () -> tensor<2x4xi64>
      %15 = "stablehlo.constant"() <{value = dense<[[9, 10, 11, 12], [13, 14, 15, 16]]> : tensor<2x4xi64>}> : () -> tensor<2x4xi64>
      %16:2 = "interpreter.run_parallel"(%14, %15) <{programs = [[@all_to_all], [@all_to_all]]}> : (tensor<2x4xi64>, tensor<2x4xi64>) -> (tensor<4x2xi64>, tensor<4x2xi64>)
      "check.expect_eq_const"(%16#0) <{value = dense<[[11, 12], [15, 16], [3, 4], [7, 8]]> : tensor<4x2xi64>}> : (tensor<4x2xi64>) -> ()
      "check.expect_eq_const"(%16#1) <{value = dense<[[9, 10], [13, 14], [1, 2], [5, 6]]> : tensor<4x2xi64>}> : (tensor<4x2xi64>) -> ()
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() <{sym_name = "cross_partition"}> ({
    "func.func"() <{function_type = (tensor<2x4xi64>) -> tensor<4x2xi64>, sym_name = "all_to_all"}> ({
    ^bb0(%arg3: tensor<2x4xi64>):
      %13 = "stablehlo.all_to_all"(%arg3) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, split_count = 2 : i64, split_dimension = 1 : i64}> : (tensor<2x4xi64>) -> tensor<4x2xi64>
      "func.return"(%13) : (tensor<4x2xi64>) -> ()
    }) : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
      %10 = "stablehlo.constant"() <{value = dense<[[1, 2, 3, 4], [5, 6, 7, 8]]> : tensor<2x4xi64>}> : () -> tensor<2x4xi64>
      %11 = "stablehlo.constant"() <{value = dense<[[9, 10, 11, 12], [13, 14, 15, 16]]> : tensor<2x4xi64>}> : () -> tensor<2x4xi64>
      %12:2 = "interpreter.run_parallel"(%10, %11) <{programs = [[@all_to_all, @all_to_all]]}> : (tensor<2x4xi64>, tensor<2x4xi64>) -> (tensor<4x2xi64>, tensor<4x2xi64>)
      "check.expect_eq_const"(%12#0) <{value = dense<[[1, 2], [5, 6], [9, 10], [13, 14]]> : tensor<4x2xi64>}> : (tensor<4x2xi64>) -> ()
      "check.expect_eq_const"(%12#1) <{value = dense<[[3, 4], [7, 8], [11, 12], [15, 16]]> : tensor<4x2xi64>}> : (tensor<4x2xi64>) -> ()
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() <{sym_name = "same_split_concat_dim"}> ({
    "func.func"() <{function_type = (tensor<2x4xi64>) -> tensor<2x4xi64>, sym_name = "all_to_all"}> ({
    ^bb0(%arg2: tensor<2x4xi64>):
      %9 = "stablehlo.all_to_all"(%arg2) <{concat_dimension = 0 : i64, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, split_count = 2 : i64, split_dimension = 0 : i64}> : (tensor<2x4xi64>) -> tensor<2x4xi64>
      "func.return"(%9) : (tensor<2x4xi64>) -> ()
    }) : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
      %6 = "stablehlo.constant"() <{value = dense<[[1, 2, 3, 4], [5, 6, 7, 8]]> : tensor<2x4xi64>}> : () -> tensor<2x4xi64>
      %7 = "stablehlo.constant"() <{value = dense<[[9, 10, 11, 12], [13, 14, 15, 16]]> : tensor<2x4xi64>}> : () -> tensor<2x4xi64>
      %8:2 = "interpreter.run_parallel"(%6, %7) <{programs = [[@all_to_all], [@all_to_all]]}> : (tensor<2x4xi64>, tensor<2x4xi64>) -> (tensor<2x4xi64>, tensor<2x4xi64>)
      "check.expect_eq_const"(%8#0) <{value = dense<[[1, 2, 3, 4], [9, 10, 11, 12]]> : tensor<2x4xi64>}> : (tensor<2x4xi64>) -> ()
      "check.expect_eq_const"(%8#1) <{value = dense<[[5, 6, 7, 8], [13, 14, 15, 16]]> : tensor<2x4xi64>}> : (tensor<2x4xi64>) -> ()
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() <{sym_name = "cross_replica_variaidic"}> ({
    "func.func"() <{function_type = (tensor<2x4xi64>, tensor<3x4xi32>) -> (tensor<4x2xi64>, tensor<6x2xi32>), sym_name = "all_to_all"}> ({
    ^bb0(%arg0: tensor<2x4xi64>, %arg1: tensor<3x4xi32>):
      %5:2 = "stablehlo.all_to_all"(%arg0, %arg1) <{concat_dimension = 0 : i64, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, split_count = 2 : i64, split_dimension = 1 : i64}> : (tensor<2x4xi64>, tensor<3x4xi32>) -> (tensor<4x2xi64>, tensor<6x2xi32>)
      "func.return"(%5#0, %5#1) : (tensor<4x2xi64>, tensor<6x2xi32>) -> ()
    }) : () -> ()
    "func.func"() <{function_type = () -> (tensor<4x2xi64>, tensor<6x2xi32>, tensor<4x2xi64>, tensor<6x2xi32>), sym_name = "main"}> ({
      %0 = "stablehlo.constant"() <{value = dense<[[1, 2, 3, 4], [5, 6, 7, 8]]> : tensor<2x4xi64>}> : () -> tensor<2x4xi64>
      %1 = "stablehlo.constant"() <{value = dense<[[9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]> : tensor<3x4xi32>}> : () -> tensor<3x4xi32>
      %2 = "stablehlo.constant"() <{value = dense<[[31, 32, 33, 34], [35, 36, 37, 38]]> : tensor<2x4xi64>}> : () -> tensor<2x4xi64>
      %3 = "stablehlo.constant"() <{value = dense<[[43, 44, 45, 46], [49, 50, 51, 52], [53, 54, 55, 56]]> : tensor<3x4xi32>}> : () -> tensor<3x4xi32>
      %4:4 = "interpreter.run_parallel"(%0, %1, %2, %3) <{programs = [[@all_to_all], [@all_to_all]]}> : (tensor<2x4xi64>, tensor<3x4xi32>, tensor<2x4xi64>, tensor<3x4xi32>) -> (tensor<4x2xi64>, tensor<6x2xi32>, tensor<4x2xi64>, tensor<6x2xi32>)
      "check.expect_eq_const"(%4#0) <{value = dense<[[1, 2], [5, 6], [31, 32], [35, 36]]> : tensor<4x2xi64>}> : (tensor<4x2xi64>) -> ()
      "check.expect_eq_const"(%4#1) <{value = dense<[[9, 10], [13, 14], [17, 18], [43, 44], [49, 50], [53, 54]]> : tensor<6x2xi32>}> : (tensor<6x2xi32>) -> ()
      "check.expect_eq_const"(%4#2) <{value = dense<[[3, 4], [7, 8], [33, 34], [37, 38]]> : tensor<4x2xi64>}> : (tensor<4x2xi64>) -> ()
      "check.expect_eq_const"(%4#3) <{value = dense<[[11, 12], [15, 16], [19, 20], [45, 46], [51, 52], [55, 56]]> : tensor<6x2xi32>}> : (tensor<6x2xi32>) -> ()
      "func.return"(%4#0, %4#1, %4#2, %4#3) : (tensor<4x2xi64>, tensor<6x2xi32>, tensor<4x2xi64>, tensor<6x2xi32>) -> ()
    }) : () -> ()
  }) : () -> ()
}) : () -> ()

