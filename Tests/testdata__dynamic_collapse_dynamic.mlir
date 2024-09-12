"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {}, {}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<i64>, tensor<i64>, tensor<?x?x5x?x7xf32>) -> tensor<?x?x7xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<i64>, %arg3: tensor<?x?x5x?x7xf32>):
    %0 = "stablehlo.multiply"(%arg1, %arg2) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %1 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
    %2 = "stablehlo.multiply"(%0, %1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %3 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %4 = "stablehlo.reshape"(%3) : (tensor<i32>) -> tensor<1xi32>
    %5 = "stablehlo.convert"(%2) : (tensor<i64>) -> tensor<i32>
    %6 = "stablehlo.reshape"(%5) : (tensor<i32>) -> tensor<1xi32>
    %7 = "stablehlo.constant"() <{value = dense<7> : tensor<1xi32>}> : () -> tensor<1xi32>
    %8 = "stablehlo.concatenate"(%4, %6, %7) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %9 = "stablehlo.dynamic_reshape"(%arg3, %8) : (tensor<?x?x5x?x7xf32>, tensor<3xi32>) -> tensor<?x?x7xf32>
    "func.return"(%9) : (tensor<?x?x7xf32>) -> ()
  }) : () -> ()
}) : () -> ()

