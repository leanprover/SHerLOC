"builtin.module"() ({
  "func.func"() <{function_type = (tensor<i64>) -> tensor<i64>, sym_name = "add_n.impl"}> ({
  ^bb0(%arg0: tensor<i64>):
    %2 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
    %3 = "stablehlo.add"(%arg0, %2) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    "func.return"(%3) : (tensor<i64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %1 = "stablehlo.composite"(%0) <{composite_attributes = {n = 2 : i64}, decomposition = @add_n.impl, name = "stablehlo.add_n"}> : (tensor<i64>) -> tensor<i64>
    "check.expect_eq_const"(%1) <{value = dense<3> : tensor<i64>}> : (tensor<i64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

