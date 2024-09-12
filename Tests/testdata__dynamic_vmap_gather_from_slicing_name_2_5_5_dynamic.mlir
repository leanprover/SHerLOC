"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x10x10x10xf32>) -> tensor<?x3x10xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x10x10x10xf32>):
    %0 = "stablehlo.constant"() <{value = dense<2> : tensor<i32>}> : () -> tensor<i32>
    %1 = "stablehlo.broadcast_in_dim"(%0) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<1xi32>
    %2 = "stablehlo.constant"() <{value = dense<5> : tensor<i32>}> : () -> tensor<i32>
    %3 = "stablehlo.broadcast_in_dim"(%2) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<1xi32>
    %4 = "stablehlo.concatenate"(%1, %3) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %5 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %6 = "stablehlo.reshape"(%5) : (tensor<i32>) -> tensor<1xi32>
    %7 = "stablehlo.constant"() <{value = dense<3> : tensor<1xi32>}> : () -> tensor<1xi32>
    %8 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %9 = "stablehlo.constant"() <{value = dense<10> : tensor<1xi32>}> : () -> tensor<1xi32>
    %10 = "stablehlo.concatenate"(%6, %7, %8, %9) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %11 = "stablehlo.dynamic_gather"(%arg1, %4, %10) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [2], start_index_map = [1, 2]>, indices_are_sorted = true}> : (tensor<?x10x10x10xf32>, tensor<2xi32>, tensor<4xi32>) -> tensor<?x3x10xf32>
    "func.return"(%11) : (tensor<?x3x10xf32>) -> ()
  }) : () -> ()
}) : () -> ()

