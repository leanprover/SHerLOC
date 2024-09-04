"builtin.module"() <{sym_name = "distribution_ops"}> ({
  "func.func"() <{function_type = (tensor<2x2x2xi64>, !stablehlo.token) -> !stablehlo.token, sym_name = "outfeed"}> ({
  ^bb0(%arg0: tensor<2x2x2xi64>, %arg1: !stablehlo.token):
    %3 = "stablehlo.outfeed"(%arg0, %arg1) <{outfeed_config = ""}> : (tensor<2x2x2xi64>, !stablehlo.token) -> !stablehlo.token
    "func.return"(%3) : (!stablehlo.token) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %1 = "stablehlo.after_all"() : () -> !stablehlo.token
    %2 = "interpreter.run_parallel"(%0, %1) <{programs = [[@outfeed]]}> : (tensor<2x2x2xi64>, !stablehlo.token) -> !stablehlo.token
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

