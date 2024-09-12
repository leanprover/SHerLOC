"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "if_ops_true_branch"}> ({
    %0 = "stablehlo.constant"() <{value = dense<true> : tensor<i1>}> : () -> tensor<i1>
    %1:2 = "stablehlo.if"(%0) ({
      %3 = "stablehlo.constant"() <{value = dense<0> : tensor<2xi64>}> : () -> tensor<2xi64>
      "stablehlo.return"(%3, %3) : (tensor<2xi64>, tensor<2xi64>) -> ()
    }, {
      %2 = "stablehlo.constant"() <{value = dense<1> : tensor<2xi64>}> : () -> tensor<2xi64>
      "stablehlo.return"(%2, %2) : (tensor<2xi64>, tensor<2xi64>) -> ()
    }) : (tensor<i1>) -> (tensor<2xi64>, tensor<2xi64>)
    "check.expect_eq_const"(%1#0) <{value = dense<0> : tensor<2xi64>}> : (tensor<2xi64>) -> ()
    "check.expect_eq_const"(%1#1) <{value = dense<0> : tensor<2xi64>}> : (tensor<2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "if_ops_false_branch"}> ({
    %0 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
    %1:2 = "stablehlo.if"(%0) ({
      %3 = "stablehlo.constant"() <{value = dense<0> : tensor<2xi64>}> : () -> tensor<2xi64>
      "stablehlo.return"(%3, %3) : (tensor<2xi64>, tensor<2xi64>) -> ()
    }, {
      %2 = "stablehlo.constant"() <{value = dense<1> : tensor<2xi64>}> : () -> tensor<2xi64>
      "stablehlo.return"(%2, %2) : (tensor<2xi64>, tensor<2xi64>) -> ()
    }) : (tensor<i1>) -> (tensor<2xi64>, tensor<2xi64>)
    "check.expect_eq_const"(%1#0) <{value = dense<1> : tensor<2xi64>}> : (tensor<2xi64>) -> ()
    "check.expect_eq_const"(%1#1) <{value = dense<1> : tensor<2xi64>}> : (tensor<2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

