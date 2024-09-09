"builtin.module"() <{sym_name = "distribution_ops"}> ({
  "func.func"() <{function_type = (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token, tensor<2x2xi64>, !stablehlo.token), sym_name = "infeed"}> ({
  ^bb0(%arg0: !stablehlo.token):
    %4:2 = "stablehlo.infeed"(%arg0) <{infeed_config = ""}> : (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token)
    %5:2 = "stablehlo.infeed"(%arg0) <{infeed_config = ""}> : (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token)
    "func.return"(%4#0, %4#1, %5#0, %5#1) : (tensor<2x2xi64>, !stablehlo.token, tensor<2x2xi64>, !stablehlo.token) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x2xi64>, sym_name = "infeed_queue0"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    "func.return"(%3) : (tensor<2x2xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x2xi64>, sym_name = "infeed_queue1"}> ({
    %2 = "stablehlo.constant"() <{value = dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    "func.return"(%2) : (tensor<2x2xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0 = "stablehlo.after_all"() : () -> !stablehlo.token
    %1:4 = "interpreter.run_parallel"(%0) <{infeed = [@infeed_queue0, @infeed_queue1], programs = [[@infeed]]}> : (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token, tensor<2x2xi64>, !stablehlo.token)
    "check.expect_eq_const"(%1#0) <{value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>}> : (tensor<2x2xi64>) -> ()
    "check.expect_eq_const"(%1#2) <{value = dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>}> : (tensor<2x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

