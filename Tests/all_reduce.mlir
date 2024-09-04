"builtin.module"() ({
  "builtin.module"() <{sym_name = "cross_replica"}> ({
    "func.func"() <{function_type = (tensor<4xi64>) -> tensor<4xi64>, sym_name = "all_reduce"}> ({
    ^bb0(%arg13: tensor<4xi64>):
      %26 = "stablehlo.all_reduce"(%arg13) <{channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}> ({
      ^bb0(%arg14: tensor<i64>, %arg15: tensor<i64>):
        %27 = "stablehlo.add"(%arg14, %arg15) : (tensor<i64>, tensor<i64>) -> tensor<i64>
        "stablehlo.return"(%27) : (tensor<i64>) -> ()
      }) : (tensor<4xi64>) -> tensor<4xi64>
      "func.return"(%26) : (tensor<4xi64>) -> ()
    }) : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
      %23 = "stablehlo.constant"() <{value = dense<[1, 2, 3, 4]> : tensor<4xi64>}> : () -> tensor<4xi64>
      %24 = "stablehlo.constant"() <{value = dense<[5, 6, 7, 8]> : tensor<4xi64>}> : () -> tensor<4xi64>
      %25:2 = "interpreter.run_parallel"(%23, %24) <{programs = [[@all_reduce], [@all_reduce]]}> : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>, tensor<4xi64>)
      "check.expect_eq_const"(%25#0) <{value = dense<[6, 8, 10, 12]> : tensor<4xi64>}> : (tensor<4xi64>) -> ()
      "check.expect_eq_const"(%25#1) <{value = dense<[6, 8, 10, 12]> : tensor<4xi64>}> : (tensor<4xi64>) -> ()
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() <{sym_name = "cross_replica_and_partition"}> ({
    "func.func"() <{function_type = (tensor<4xi64>) -> tensor<4xi64>, sym_name = "all_reduce"}> ({
    ^bb0(%arg10: tensor<4xi64>):
      %21 = "stablehlo.all_reduce"(%arg10) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}> ({
      ^bb0(%arg11: tensor<i64>, %arg12: tensor<i64>):
        %22 = "stablehlo.add"(%arg11, %arg12) : (tensor<i64>, tensor<i64>) -> tensor<i64>
        "stablehlo.return"(%22) : (tensor<i64>) -> ()
      }) : (tensor<4xi64>) -> tensor<4xi64>
      "func.return"(%21) : (tensor<4xi64>) -> ()
    }) : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
      %18 = "stablehlo.constant"() <{value = dense<[1, 2, 3, 4]> : tensor<4xi64>}> : () -> tensor<4xi64>
      %19 = "stablehlo.constant"() <{value = dense<[5, 6, 7, 8]> : tensor<4xi64>}> : () -> tensor<4xi64>
      %20:2 = "interpreter.run_parallel"(%18, %19) <{programs = [[@all_reduce], [@all_reduce]]}> : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>, tensor<4xi64>)
      "check.expect_eq_const"(%20#0) <{value = dense<[6, 8, 10, 12]> : tensor<4xi64>}> : (tensor<4xi64>) -> ()
      "check.expect_eq_const"(%20#1) <{value = dense<[6, 8, 10, 12]> : tensor<4xi64>}> : (tensor<4xi64>) -> ()
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() <{sym_name = "flattened_ids"}> ({
    "func.func"() <{function_type = (tensor<4xi64>) -> tensor<4xi64>, sym_name = "all_reduce"}> ({
    ^bb0(%arg7: tensor<4xi64>):
      %16 = "stablehlo.all_reduce"(%arg7) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> ({
      ^bb0(%arg8: tensor<i64>, %arg9: tensor<i64>):
        %17 = "stablehlo.add"(%arg8, %arg9) : (tensor<i64>, tensor<i64>) -> tensor<i64>
        "stablehlo.return"(%17) : (tensor<i64>) -> ()
      }) : (tensor<4xi64>) -> tensor<4xi64>
      "func.return"(%16) : (tensor<4xi64>) -> ()
    }) : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
      %13 = "stablehlo.constant"() <{value = dense<[1, 2, 3, 4]> : tensor<4xi64>}> : () -> tensor<4xi64>
      %14 = "stablehlo.constant"() <{value = dense<[5, 6, 7, 8]> : tensor<4xi64>}> : () -> tensor<4xi64>
      %15:2 = "interpreter.run_parallel"(%13, %14) <{programs = [[@all_reduce], [@all_reduce]]}> : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>, tensor<4xi64>)
      "check.expect_eq_const"(%15#0) <{value = dense<[6, 8, 10, 12]> : tensor<4xi64>}> : (tensor<4xi64>) -> ()
      "check.expect_eq_const"(%15#1) <{value = dense<[6, 8, 10, 12]> : tensor<4xi64>}> : (tensor<4xi64>) -> ()
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() <{sym_name = "ragged_replica_groups"}> ({
    "func.func"() <{function_type = (tensor<4xi64>) -> tensor<4xi64>, sym_name = "all_reduce"}> ({
    ^bb0(%arg4: tensor<4xi64>):
      %11 = "stablehlo.all_reduce"(%arg4) <{channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>, replica_groups = dense<[[0, 1], [2, -1]]> : tensor<2x2xi64>}> ({
      ^bb0(%arg5: tensor<i64>, %arg6: tensor<i64>):
        %12 = "stablehlo.add"(%arg5, %arg6) : (tensor<i64>, tensor<i64>) -> tensor<i64>
        "stablehlo.return"(%12) : (tensor<i64>) -> ()
      }) : (tensor<4xi64>) -> tensor<4xi64>
      "func.return"(%11) : (tensor<4xi64>) -> ()
    }) : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
      %7 = "stablehlo.constant"() <{value = dense<[1, 2, 3, 4]> : tensor<4xi64>}> : () -> tensor<4xi64>
      %8 = "stablehlo.constant"() <{value = dense<[5, 6, 7, 8]> : tensor<4xi64>}> : () -> tensor<4xi64>
      %9 = "stablehlo.constant"() <{value = dense<[6, 8, 10, 12]> : tensor<4xi64>}> : () -> tensor<4xi64>
      %10:3 = "interpreter.run_parallel"(%7, %8, %9) <{programs = [[@all_reduce], [@all_reduce], [@all_reduce]]}> : (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>)
      "check.expect_eq_const"(%10#0) <{value = dense<[6, 8, 10, 12]> : tensor<4xi64>}> : (tensor<4xi64>) -> ()
      "check.expect_eq_const"(%10#1) <{value = dense<[6, 8, 10, 12]> : tensor<4xi64>}> : (tensor<4xi64>) -> ()
      "check.expect_eq_const"(%10#2) <{value = dense<[6, 8, 10, 12]> : tensor<4xi64>}> : (tensor<4xi64>) -> ()
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() <{sym_name = "cross_replica_variadic"}> ({
    "func.func"() <{function_type = (tensor<4xi64>, tensor<5xi64>) -> (tensor<4xi64>, tensor<5xi64>), sym_name = "all_reduce"}> ({
    ^bb0(%arg0: tensor<4xi64>, %arg1: tensor<5xi64>):
      %5:2 = "stablehlo.all_reduce"(%arg0, %arg1) <{channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}> ({
      ^bb0(%arg2: tensor<i64>, %arg3: tensor<i64>):
        %6 = "stablehlo.add"(%arg2, %arg3) : (tensor<i64>, tensor<i64>) -> tensor<i64>
        "stablehlo.return"(%6) : (tensor<i64>) -> ()
      }) : (tensor<4xi64>, tensor<5xi64>) -> (tensor<4xi64>, tensor<5xi64>)
      "func.return"(%5#0, %5#1) : (tensor<4xi64>, tensor<5xi64>) -> ()
    }) : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
      %0 = "stablehlo.constant"() <{value = dense<[1, 2, 3, 4]> : tensor<4xi64>}> : () -> tensor<4xi64>
      %1 = "stablehlo.constant"() <{value = dense<[5, 6, 7, 8, 9]> : tensor<5xi64>}> : () -> tensor<5xi64>
      %2 = "stablehlo.constant"() <{value = dense<[11, 12, 13, 14]> : tensor<4xi64>}> : () -> tensor<4xi64>
      %3 = "stablehlo.constant"() <{value = dense<[15, 16, 17, 18, 19]> : tensor<5xi64>}> : () -> tensor<5xi64>
      %4:4 = "interpreter.run_parallel"(%0, %1, %2, %3) <{programs = [[@all_reduce], [@all_reduce]]}> : (tensor<4xi64>, tensor<5xi64>, tensor<4xi64>, tensor<5xi64>) -> (tensor<4xi64>, tensor<5xi64>, tensor<4xi64>, tensor<5xi64>)
      "check.expect_eq_const"(%4#0) <{value = dense<[12, 14, 16, 18]> : tensor<4xi64>}> : (tensor<4xi64>) -> ()
      "check.expect_eq_const"(%4#1) <{value = dense<[20, 22, 24, 26, 28]> : tensor<5xi64>}> : (tensor<5xi64>) -> ()
      "check.expect_eq_const"(%4#2) <{value = dense<[12, 14, 16, 18]> : tensor<4xi64>}> : (tensor<4xi64>) -> ()
      "check.expect_eq_const"(%4#3) <{value = dense<[20, 22, 24, 26, 28]> : tensor<5xi64>}> : (tensor<5xi64>) -> ()
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
}) : () -> ()

