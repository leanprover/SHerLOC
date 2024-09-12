"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x3x4x5xf32>) -> tensor<?x3x20xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x3x4x5xf32>):
    %0 = "stablehlo.transpose"(%arg1) <{permutation = array<i64: 0, 3, 2, 1>}> : (tensor<?x3x4x5xf32>) -> tensor<?x5x4x3xf32>
    %1 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %2 = "stablehlo.reshape"(%1) : (tensor<i32>) -> tensor<1xi32>
    %3 = "stablehlo.constant"() <{value = dense<3> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "stablehlo.constant"() <{value = dense<20> : tensor<1xi32>}> : () -> tensor<1xi32>
    %5 = "stablehlo.concatenate"(%2, %3, %4) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %6 = "stablehlo.dynamic_reshape"(%0, %5) : (tensor<?x5x4x3xf32>, tensor<3xi32>) -> tensor<?x3x20xf32>
    "func.return"(%6) : (tensor<?x3x20xf32>) -> ()
  }) : () -> ()
}) : () -> ()

