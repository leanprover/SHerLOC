"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "case_negative_index_default"}> ({
    %0 = "stablehlo.constant"() <{value = dense<-1> : tensor<i32>}> : () -> tensor<i32>
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<2xi64>}> : () -> tensor<2xi64>
    %2 = "stablehlo.constant"() <{value = dense<1> : tensor<2xi64>}> : () -> tensor<2xi64>
    %3:2 = "stablehlo.case"(%0) ({
      "stablehlo.return"(%1, %1) : (tensor<2xi64>, tensor<2xi64>) -> ()
    }, {
      "stablehlo.return"(%2, %2) : (tensor<2xi64>, tensor<2xi64>) -> ()
    }) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)
    "check.expect_eq_const"(%3#0) <{value = dense<1> : tensor<2xi64>}> : (tensor<2xi64>) -> ()
    "check.expect_eq_const"(%3#1) <{value = dense<1> : tensor<2xi64>}> : (tensor<2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "case_in_bound_index"}> ({
    %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<2xi64>}> : () -> tensor<2xi64>
    %2 = "stablehlo.constant"() <{value = dense<1> : tensor<2xi64>}> : () -> tensor<2xi64>
    %3:2 = "stablehlo.case"(%0) ({
      "stablehlo.return"(%1, %1) : (tensor<2xi64>, tensor<2xi64>) -> ()
    }, {
      "stablehlo.return"(%2, %2) : (tensor<2xi64>, tensor<2xi64>) -> ()
    }) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)
    "check.expect_eq_const"(%3#0) <{value = dense<0> : tensor<2xi64>}> : (tensor<2xi64>) -> ()
    "check.expect_eq_const"(%3#1) <{value = dense<0> : tensor<2xi64>}> : (tensor<2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "case_out_of_bound_index_default"}> ({
    %0 = "stablehlo.constant"() <{value = dense<2> : tensor<i32>}> : () -> tensor<i32>
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<2xi64>}> : () -> tensor<2xi64>
    %2 = "stablehlo.constant"() <{value = dense<1> : tensor<2xi64>}> : () -> tensor<2xi64>
    %3:2 = "stablehlo.case"(%0) ({
      "stablehlo.return"(%1, %1) : (tensor<2xi64>, tensor<2xi64>) -> ()
    }, {
      "stablehlo.return"(%2, %2) : (tensor<2xi64>, tensor<2xi64>) -> ()
    }) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)
    "check.expect_eq_const"(%3#0) <{value = dense<1> : tensor<2xi64>}> : (tensor<2xi64>) -> ()
    "check.expect_eq_const"(%3#1) <{value = dense<1> : tensor<2xi64>}> : (tensor<2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

