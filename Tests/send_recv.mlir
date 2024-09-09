"builtin.module"() ({
  "builtin.module"() <{sym_name = "sequential_send_recv_same_channel"}> ({
    "func.func"() <{function_type = (tensor<2x2xi64>, !stablehlo.token) -> (!stablehlo.token, !stablehlo.token), sym_name = "send"}> ({
    ^bb0(%arg7: tensor<2x2xi64>, %arg8: !stablehlo.token):
      %13 = "stablehlo.send"(%arg7, %arg8) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 2>, is_host_transfer = true}> : (tensor<2x2xi64>, !stablehlo.token) -> !stablehlo.token
      %14 = "stablehlo.constant"() <{value = dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
      %15 = "stablehlo.send"(%14, %arg8) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 2>, is_host_transfer = true}> : (tensor<2x2xi64>, !stablehlo.token) -> !stablehlo.token
      "func.return"(%13, %15) : (!stablehlo.token, !stablehlo.token) -> ()
    }) : () -> ()
    "func.func"() <{function_type = (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token, tensor<2x2xi64>, !stablehlo.token), sym_name = "recv"}> ({
    ^bb0(%arg6: !stablehlo.token):
      %11:2 = "stablehlo.recv"(%arg6) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 3>, is_host_transfer = true}> : (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token)
      %12:2 = "stablehlo.recv"(%arg6) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 3>, is_host_transfer = true}> : (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token)
      "func.return"(%11#0, %11#1, %12#0, %12#1) : (tensor<2x2xi64>, !stablehlo.token, tensor<2x2xi64>, !stablehlo.token) -> ()
    }) : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
      %8 = "stablehlo.constant"() <{value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
      %9 = "stablehlo.after_all"() : () -> !stablehlo.token
      %10:6 = "interpreter.run_parallel"(%8, %9, %9) <{programs = [[@send], [@recv]]}> : (tensor<2x2xi64>, !stablehlo.token, !stablehlo.token) -> (!stablehlo.token, !stablehlo.token, tensor<2x2xi64>, !stablehlo.token, tensor<2x2xi64>, !stablehlo.token)
      "check.expect_eq_const"(%10#2) <{value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>}> : (tensor<2x2xi64>) -> ()
      "check.expect_eq_const"(%10#4) <{value = dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>}> : (tensor<2x2xi64>) -> ()
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() <{sym_name = "paralllel_send_recv_different_channels"}> ({
    "func.func"() <{function_type = (tensor<2x2xi64>, !stablehlo.token) -> !stablehlo.token, sym_name = "send0"}> ({
    ^bb0(%arg4: tensor<2x2xi64>, %arg5: !stablehlo.token):
      %7 = "stablehlo.send"(%arg4, %arg5) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 2>, is_host_transfer = true}> : (tensor<2x2xi64>, !stablehlo.token) -> !stablehlo.token
      "func.return"(%7) : (!stablehlo.token) -> ()
    }) : () -> ()
    "func.func"() <{function_type = (tensor<2x2xi64>, !stablehlo.token) -> !stablehlo.token, sym_name = "send1"}> ({
    ^bb0(%arg2: tensor<2x2xi64>, %arg3: !stablehlo.token):
      %6 = "stablehlo.send"(%arg2, %arg3) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 2>, is_host_transfer = true}> : (tensor<2x2xi64>, !stablehlo.token) -> !stablehlo.token
      "func.return"(%6) : (!stablehlo.token) -> ()
    }) : () -> ()
    "func.func"() <{function_type = (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token), sym_name = "recv0"}> ({
    ^bb0(%arg1: !stablehlo.token):
      %5:2 = "stablehlo.recv"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 3>, is_host_transfer = true}> : (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token)
      "func.return"(%5#0, %5#1) : (tensor<2x2xi64>, !stablehlo.token) -> ()
    }) : () -> ()
    "func.func"() <{function_type = (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token), sym_name = "recv1"}> ({
    ^bb0(%arg0: !stablehlo.token):
      %4:2 = "stablehlo.recv"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 3>, is_host_transfer = true}> : (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token)
      "func.return"(%4#0, %4#1) : (tensor<2x2xi64>, !stablehlo.token) -> ()
    }) : () -> ()
    "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
      %0 = "stablehlo.constant"() <{value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
      %1 = "stablehlo.constant"() <{value = dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
      %2 = "stablehlo.after_all"() : () -> !stablehlo.token
      %3:6 = "interpreter.run_parallel"(%0, %2, %2, %1, %2, %2) <{programs = [[@send0], [@recv0], [@send1], [@recv1]]}> : (tensor<2x2xi64>, !stablehlo.token, !stablehlo.token, tensor<2x2xi64>, !stablehlo.token, !stablehlo.token) -> (!stablehlo.token, tensor<2x2xi64>, !stablehlo.token, !stablehlo.token, tensor<2x2xi64>, !stablehlo.token)
      "check.expect_eq_const"(%3#1) <{value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>}> : (tensor<2x2xi64>) -> ()
      "check.expect_eq_const"(%3#4) <{value = dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>}> : (tensor<2x2xi64>) -> ()
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
}) : () -> ()

