"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "optimization_barrier_op_test"}> ({
    %0 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %1 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %2:2 = "stablehlo.optimization_barrier"(%0, %1) : (tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
    "check.expect_almost_eq_const"(%2#0) <{value = dense<0.000000e+00> : tensor<f32>}> : (tensor<f32>) -> ()
    "check.expect_almost_eq_const"(%2#1) <{value = dense<1.000000e+00> : tensor<f32>}> : (tensor<f32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

