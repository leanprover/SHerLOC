"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "case_negative_index_default"}> ({
    %8 = "stablehlo.constant"() <{value = dense<-1> : tensor<i32>}> : () -> tensor<i32>
    %9 = "stablehlo.constant"() <{value = dense<0> : tensor<2xi64>}> : () -> tensor<2xi64>
    %10 = "stablehlo.constant"() <{value = dense<1> : tensor<2xi64>}> : () -> tensor<2xi64>
    %11:2 = "stablehlo.case"(%8) ({
      "stablehlo.return"(%9, %9) : (tensor<2xi64>, tensor<2xi64>) -> ()
    }, {
      "stablehlo.return"(%10, %10) : (tensor<2xi64>, tensor<2xi64>) -> ()
    }) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)
    "check.expect_eq_const"(%11#0) <{value = dense<1> : tensor<2xi64>}> : (tensor<2xi64>) -> ()
    "check.expect_eq_const"(%11#1) <{value = dense<1> : tensor<2xi64>}> : (tensor<2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "case_in_bound_index"}> ({
    %4 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %5 = "stablehlo.constant"() <{value = dense<0> : tensor<2xi64>}> : () -> tensor<2xi64>
    %6 = "stablehlo.constant"() <{value = dense<1> : tensor<2xi64>}> : () -> tensor<2xi64>
    %7:2 = "stablehlo.case"(%4) ({
      "stablehlo.return"(%5, %5) : (tensor<2xi64>, tensor<2xi64>) -> ()
    }, {
      "stablehlo.return"(%6, %6) : (tensor<2xi64>, tensor<2xi64>) -> ()
    }) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)
    "check.expect_eq_const"(%7#0) <{value = dense<0> : tensor<2xi64>}> : (tensor<2xi64>) -> ()
    "check.expect_eq_const"(%7#1) <{value = dense<0> : tensor<2xi64>}> : (tensor<2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
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

