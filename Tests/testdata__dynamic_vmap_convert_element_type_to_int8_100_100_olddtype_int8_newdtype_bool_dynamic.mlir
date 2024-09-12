"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x100x100xi8>) -> tensor<?x100x100xi1>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x100x100xi8>):
    %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i8>}> : () -> tensor<i8>
    %1 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %2 = "stablehlo.reshape"(%1) : (tensor<i32>) -> tensor<1xi32>
    %3 = "stablehlo.constant"() <{value = dense<100> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "stablehlo.constant"() <{value = dense<100> : tensor<1xi32>}> : () -> tensor<1xi32>
    %5 = "stablehlo.concatenate"(%2, %3, %4) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %6 = "stablehlo.dynamic_broadcast_in_dim"(%0, %5) <{broadcast_dimensions = array<i64>}> : (tensor<i8>, tensor<3xi32>) -> tensor<?x100x100xi8>
    %7 = "stablehlo.compare"(%arg1, %6) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<?x100x100xi8>, tensor<?x100x100xi8>) -> tensor<?x100x100xi1>
    "func.return"(%7) : (tensor<?x100x100xi1>) -> ()
  }) : () -> ()
}) : () -> ()

