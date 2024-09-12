"builtin.module"() <{sym_name = "cross_replica"}> ({
  "func.func"() <{function_type = (tensor<4xi64>) -> tensor<4xi64>, sym_name = "all_reduce"}> ({
  ^bb0(%arg0: tensor<4xi64>):
    %3 = "stablehlo.all_reduce"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}> ({
    ^bb0(%arg1: tensor<i64>, %arg2: tensor<i64>):
      %4 = "stablehlo.add"(%arg1, %arg2) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%4) : (tensor<i64>) -> ()
    }) : (tensor<4xi64>) -> tensor<4xi64>
    "func.return"(%3) : (tensor<4xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[1, 2, 3, 4]> : tensor<4xi64>}> : () -> tensor<4xi64>
    %1 = "stablehlo.constant"() <{value = dense<[5, 6, 7, 8]> : tensor<4xi64>}> : () -> tensor<4xi64>
    %2:2 = "interpreter.run_parallel"(%0, %1) <{programs = [[@all_reduce], [@all_reduce]]}> : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>, tensor<4xi64>)
    "check.expect_eq_const"(%2#0) <{value = dense<[6, 8, 10, 12]> : tensor<4xi64>}> : (tensor<4xi64>) -> ()
    "check.expect_eq_const"(%2#1) <{value = dense<[6, 8, 10, 12]> : tensor<4xi64>}> : (tensor<4xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() <{sym_name = "cross_replica_and_partition"}> ({
  "func.func"() <{function_type = (tensor<4xi64>) -> tensor<4xi64>, sym_name = "all_reduce"}> ({
  ^bb0(%arg0: tensor<4xi64>):
    %3 = "stablehlo.all_reduce"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}> ({
    ^bb0(%arg1: tensor<i64>, %arg2: tensor<i64>):
      %4 = "stablehlo.add"(%arg1, %arg2) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%4) : (tensor<i64>) -> ()
    }) : (tensor<4xi64>) -> tensor<4xi64>
    "func.return"(%3) : (tensor<4xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[1, 2, 3, 4]> : tensor<4xi64>}> : () -> tensor<4xi64>
    %1 = "stablehlo.constant"() <{value = dense<[5, 6, 7, 8]> : tensor<4xi64>}> : () -> tensor<4xi64>
    %2:2 = "interpreter.run_parallel"(%0, %1) <{programs = [[@all_reduce], [@all_reduce]]}> : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>, tensor<4xi64>)
    "check.expect_eq_const"(%2#0) <{value = dense<[6, 8, 10, 12]> : tensor<4xi64>}> : (tensor<4xi64>) -> ()
    "check.expect_eq_const"(%2#1) <{value = dense<[6, 8, 10, 12]> : tensor<4xi64>}> : (tensor<4xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() <{sym_name = "flattened_ids"}> ({
  "func.func"() <{function_type = (tensor<4xi64>) -> tensor<4xi64>, sym_name = "all_reduce"}> ({
  ^bb0(%arg0: tensor<4xi64>):
    %3 = "stablehlo.all_reduce"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> ({
    ^bb0(%arg1: tensor<i64>, %arg2: tensor<i64>):
      %4 = "stablehlo.add"(%arg1, %arg2) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%4) : (tensor<i64>) -> ()
    }) : (tensor<4xi64>) -> tensor<4xi64>
    "func.return"(%3) : (tensor<4xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[1, 2, 3, 4]> : tensor<4xi64>}> : () -> tensor<4xi64>
    %1 = "stablehlo.constant"() <{value = dense<[5, 6, 7, 8]> : tensor<4xi64>}> : () -> tensor<4xi64>
    %2:2 = "interpreter.run_parallel"(%0, %1) <{programs = [[@all_reduce], [@all_reduce]]}> : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>, tensor<4xi64>)
    "check.expect_eq_const"(%2#0) <{value = dense<[6, 8, 10, 12]> : tensor<4xi64>}> : (tensor<4xi64>) -> ()
    "check.expect_eq_const"(%2#1) <{value = dense<[6, 8, 10, 12]> : tensor<4xi64>}> : (tensor<4xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() <{sym_name = "ragged_replica_groups"}> ({
  "func.func"() <{function_type = (tensor<4xi64>) -> tensor<4xi64>, sym_name = "all_reduce"}> ({
  ^bb0(%arg0: tensor<4xi64>):
    %4 = "stablehlo.all_reduce"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>, replica_groups = dense<[[0, 1], [2, -1]]> : tensor<2x2xi64>}> ({
    ^bb0(%arg1: tensor<i64>, %arg2: tensor<i64>):
      %5 = "stablehlo.add"(%arg1, %arg2) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%5) : (tensor<i64>) -> ()
    }) : (tensor<4xi64>) -> tensor<4xi64>
    "func.return"(%4) : (tensor<4xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[1, 2, 3, 4]> : tensor<4xi64>}> : () -> tensor<4xi64>
    %1 = "stablehlo.constant"() <{value = dense<[5, 6, 7, 8]> : tensor<4xi64>}> : () -> tensor<4xi64>
    %2 = "stablehlo.constant"() <{value = dense<[6, 8, 10, 12]> : tensor<4xi64>}> : () -> tensor<4xi64>
    %3:3 = "interpreter.run_parallel"(%0, %1, %2) <{programs = [[@all_reduce], [@all_reduce], [@all_reduce]]}> : (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>)
    "check.expect_eq_const"(%3#0) <{value = dense<[6, 8, 10, 12]> : tensor<4xi64>}> : (tensor<4xi64>) -> ()
    "check.expect_eq_const"(%3#1) <{value = dense<[6, 8, 10, 12]> : tensor<4xi64>}> : (tensor<4xi64>) -> ()
    "check.expect_eq_const"(%3#2) <{value = dense<[6, 8, 10, 12]> : tensor<4xi64>}> : (tensor<4xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
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

