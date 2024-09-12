"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x5xf32>) -> tensor<?x1xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x5xf32>):
    %0 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %1 = "stablehlo.reduce"(%arg1, %0) <{dimensions = array<i64: 1>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %7 = "stablehlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%7) : (tensor<f32>) -> ()
    }) : (tensor<?x5xf32>, tensor<f32>) -> tensor<?xf32>
    %2 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %3 = "stablehlo.reshape"(%2) : (tensor<i32>) -> tensor<1xi32>
    %4 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %5 = "stablehlo.concatenate"(%3, %4) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %6 = "stablehlo.dynamic_broadcast_in_dim"(%1, %5) <{broadcast_dimensions = array<i64: 0>}> : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x1xf32>
    "func.return"(%6) : (tensor<?x1xf32>) -> ()
  }) : () -> ()
}) : () -> ()

