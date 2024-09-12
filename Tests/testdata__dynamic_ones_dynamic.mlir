"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x2x4xf32>) -> tensor<?x2x4xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x2x4xf32>):
    %0 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %1 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %2 = "stablehlo.reshape"(%1) : (tensor<i32>) -> tensor<1xi32>
    %3 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "stablehlo.constant"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
    %5 = "stablehlo.concatenate"(%2, %3, %4) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %6 = "stablehlo.dynamic_broadcast_in_dim"(%0, %5) <{broadcast_dimensions = array<i64>}> : (tensor<f32>, tensor<3xi32>) -> tensor<?x2x4xf32>
    %7 = "stablehlo.add"(%6, %arg1) : (tensor<?x2x4xf32>, tensor<?x2x4xf32>) -> tensor<?x2x4xf32>
    "func.return"(%7) : (tensor<?x2x4xf32>) -> ()
  }) : () -> ()
}) : () -> ()

