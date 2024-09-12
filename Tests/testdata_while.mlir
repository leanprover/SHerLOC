"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "while"}> ({
    %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %2 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %3 = "stablehlo.constant"() <{value = dense<10> : tensor<i64>}> : () -> tensor<i64>
    %4:2 = "stablehlo.while"(%0, %1) ({
    ^bb0(%arg2: tensor<i64>, %arg3: tensor<i64>):
      %7 = "stablehlo.compare"(%arg2, %3) <{comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%7) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %5 = "stablehlo.add"(%arg1, %2) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      %6 = "stablehlo.add"(%arg0, %2) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%6, %5) : (tensor<i64>, tensor<i64>) -> ()
    }) : (tensor<i64>, tensor<i64>) -> (tensor<i64>, tensor<i64>)
    "check.expect_eq_const"(%4#0) <{value = dense<10> : tensor<i64>}> : (tensor<i64>) -> ()
    "check.expect_eq_const"(%4#1) <{value = dense<10> : tensor<i64>}> : (tensor<i64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

