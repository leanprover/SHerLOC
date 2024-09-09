"builtin.module"() ({
  "builtin.module"() <{sym_name = "cross_replica"}> ({
    "func.func"() <{function_type = (tensor<1x2xi64>) -> tensor<1x2xi64>, sym_name = "collective_broadcast"}> ({
    ^bb0(%arg7: tensor<1x2xi64>):
      %47 = "stablehlo.collective_broadcast"(%arg7) <{channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>, replica_groups = dense<[[2, 1]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> tensor<1x2xi64>
      "func.return"(%47) : (tensor<1x2xi64>) -> ()
    }) : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
      %42 = "stablehlo.constant"() <{value = dense<[[1, 2]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %43 = "stablehlo.constant"() <{value = dense<[[3, 4]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %44 = "stablehlo.constant"() <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %45 = "stablehlo.constant"() <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %46:4 = "interpreter.run_parallel"(%42, %43, %44, %45) <{programs = [[@collective_broadcast], [@collective_broadcast], [@collective_broadcast], [@collective_broadcast]]}> : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>) -> (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>)
      "check.expect_eq_const"(%46#0) <{value = dense<0> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "check.expect_eq_const"(%46#1) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "check.expect_eq_const"(%46#2) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "check.expect_eq_const"(%46#3) <{value = dense<0> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() <{sym_name = "cross_replica_multiple_output"}> ({
    "func.func"() <{function_type = (tensor<1x2xi64>) -> tensor<1x2xi64>, sym_name = "collective_broadcast"}> ({
    ^bb0(%arg6: tensor<1x2xi64>):
      %41 = "stablehlo.collective_broadcast"(%arg6) <{channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>, replica_groups = dense<[[2, 1, 0]]> : tensor<1x3xi64>}> : (tensor<1x2xi64>) -> tensor<1x2xi64>
      "func.return"(%41) : (tensor<1x2xi64>) -> ()
    }) : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
      %36 = "stablehlo.constant"() <{value = dense<[[1, 2]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %37 = "stablehlo.constant"() <{value = dense<[[3, 4]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %38 = "stablehlo.constant"() <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %39 = "stablehlo.constant"() <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %40:4 = "interpreter.run_parallel"(%36, %37, %38, %39) <{programs = [[@collective_broadcast], [@collective_broadcast], [@collective_broadcast], [@collective_broadcast]]}> : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>) -> (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>)
      "check.expect_eq_const"(%40#0) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "check.expect_eq_const"(%40#1) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "check.expect_eq_const"(%40#2) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "check.expect_eq_const"(%40#3) <{value = dense<0> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() <{sym_name = "cross_replica_single_replica"}> ({
    "func.func"() <{function_type = (tensor<1x2xi64>) -> tensor<1x2xi64>, sym_name = "collective_broadcast"}> ({
    ^bb0(%arg5: tensor<1x2xi64>):
      %35 = "stablehlo.collective_broadcast"(%arg5) <{channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>, replica_groups = dense<0> : tensor<1x1xi64>}> : (tensor<1x2xi64>) -> tensor<1x2xi64>
      "func.return"(%35) : (tensor<1x2xi64>) -> ()
    }) : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
      %30 = "stablehlo.constant"() <{value = dense<[[1, 2]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %31 = "stablehlo.constant"() <{value = dense<[[3, 4]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %32 = "stablehlo.constant"() <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %33 = "stablehlo.constant"() <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %34:4 = "interpreter.run_parallel"(%30, %31, %32, %33) <{programs = [[@collective_broadcast, @collective_broadcast, @collective_broadcast, @collective_broadcast]]}> : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>) -> (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>)
      "check.expect_eq_const"(%34#0) <{value = dense<[[1, 2]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "check.expect_eq_const"(%34#1) <{value = dense<[[3, 4]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "check.expect_eq_const"(%34#2) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "check.expect_eq_const"(%34#3) <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() <{sym_name = "cross_replica_multiple_partitions"}> ({
    "func.func"() <{function_type = (tensor<1x2xi64>) -> tensor<1x2xi64>, sym_name = "collective_broadcast"}> ({
    ^bb0(%arg4: tensor<1x2xi64>):
      %29 = "stablehlo.collective_broadcast"(%arg4) <{channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>, replica_groups = dense<[[1, 0]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> tensor<1x2xi64>
      "func.return"(%29) : (tensor<1x2xi64>) -> ()
    }) : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
      %24 = "stablehlo.constant"() <{value = dense<[[1, 2]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %25 = "stablehlo.constant"() <{value = dense<[[3, 4]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %26 = "stablehlo.constant"() <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %27 = "stablehlo.constant"() <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %28:4 = "interpreter.run_parallel"(%24, %25, %26, %27) <{programs = [[@collective_broadcast, @collective_broadcast], [@collective_broadcast, @collective_broadcast]]}> : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>) -> (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>)
      "check.expect_eq_const"(%28#0) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "check.expect_eq_const"(%28#1) <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "check.expect_eq_const"(%28#2) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "check.expect_eq_const"(%28#3) <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() <{sym_name = "cross_partition"}> ({
    "func.func"() <{function_type = (tensor<1x2xi64>) -> tensor<1x2xi64>, sym_name = "collective_broadcast"}> ({
    ^bb0(%arg3: tensor<1x2xi64>):
      %23 = "stablehlo.collective_broadcast"(%arg3) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[2, 1]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> tensor<1x2xi64>
      "func.return"(%23) : (tensor<1x2xi64>) -> ()
    }) : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
      %18 = "stablehlo.constant"() <{value = dense<[[1, 2]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %19 = "stablehlo.constant"() <{value = dense<[[3, 4]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %20 = "stablehlo.constant"() <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %21 = "stablehlo.constant"() <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %22:4 = "interpreter.run_parallel"(%18, %19, %20, %21) <{programs = [[@collective_broadcast, @collective_broadcast, @collective_broadcast, @collective_broadcast]]}> : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>) -> (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>)
      "check.expect_eq_const"(%22#0) <{value = dense<0> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "check.expect_eq_const"(%22#1) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "check.expect_eq_const"(%22#2) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "check.expect_eq_const"(%22#3) <{value = dense<0> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() <{sym_name = "cross_partition_multiple_output"}> ({
    "func.func"() <{function_type = (tensor<1x2xi64>) -> tensor<1x2xi64>, sym_name = "collective_broadcast"}> ({
    ^bb0(%arg2: tensor<1x2xi64>):
      %17 = "stablehlo.collective_broadcast"(%arg2) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[2, 1, 0]]> : tensor<1x3xi64>}> : (tensor<1x2xi64>) -> tensor<1x2xi64>
      "func.return"(%17) : (tensor<1x2xi64>) -> ()
    }) : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
      %12 = "stablehlo.constant"() <{value = dense<[[1, 2]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %13 = "stablehlo.constant"() <{value = dense<[[3, 4]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %14 = "stablehlo.constant"() <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %15 = "stablehlo.constant"() <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %16:4 = "interpreter.run_parallel"(%12, %13, %14, %15) <{programs = [[@collective_broadcast, @collective_broadcast, @collective_broadcast, @collective_broadcast]]}> : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>) -> (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>)
      "check.expect_eq_const"(%16#0) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "check.expect_eq_const"(%16#1) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "check.expect_eq_const"(%16#2) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "check.expect_eq_const"(%16#3) <{value = dense<0> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() <{sym_name = "cross_partition_single_partition"}> ({
    "func.func"() <{function_type = (tensor<1x2xi64>) -> tensor<1x2xi64>, sym_name = "collective_broadcast"}> ({
    ^bb0(%arg1: tensor<1x2xi64>):
      %11 = "stablehlo.collective_broadcast"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<0> : tensor<1x1xi64>}> : (tensor<1x2xi64>) -> tensor<1x2xi64>
      "func.return"(%11) : (tensor<1x2xi64>) -> ()
    }) : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
      %6 = "stablehlo.constant"() <{value = dense<[[1, 2]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %7 = "stablehlo.constant"() <{value = dense<[[3, 4]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %8 = "stablehlo.constant"() <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %9 = "stablehlo.constant"() <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %10:4 = "interpreter.run_parallel"(%6, %7, %8, %9) <{programs = [[@collective_broadcast], [@collective_broadcast], [@collective_broadcast], [@collective_broadcast]]}> : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>) -> (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>)
      "check.expect_eq_const"(%10#0) <{value = dense<[[1, 2]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "check.expect_eq_const"(%10#1) <{value = dense<[[3, 4]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "check.expect_eq_const"(%10#2) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "check.expect_eq_const"(%10#3) <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() <{sym_name = "cross_partition_multiple_replicas"}> ({
    "func.func"() <{function_type = (tensor<1x2xi64>) -> tensor<1x2xi64>, sym_name = "collective_broadcast"}> ({
    ^bb0(%arg0: tensor<1x2xi64>):
      %5 = "stablehlo.collective_broadcast"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[1, 0]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> tensor<1x2xi64>
      "func.return"(%5) : (tensor<1x2xi64>) -> ()
    }) : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
      %0 = "stablehlo.constant"() <{value = dense<[[1, 2]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %1 = "stablehlo.constant"() <{value = dense<[[3, 4]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %2 = "stablehlo.constant"() <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %3 = "stablehlo.constant"() <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
      %4:4 = "interpreter.run_parallel"(%0, %1, %2, %3) <{programs = [[@collective_broadcast, @collective_broadcast], [@collective_broadcast, @collective_broadcast]]}> : (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>) -> (tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>, tensor<1x2xi64>)
      "check.expect_eq_const"(%4#0) <{value = dense<[[3, 4]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "check.expect_eq_const"(%4#1) <{value = dense<[[3, 4]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "check.expect_eq_const"(%4#2) <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "check.expect_eq_const"(%4#3) <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
}) : () -> ()

