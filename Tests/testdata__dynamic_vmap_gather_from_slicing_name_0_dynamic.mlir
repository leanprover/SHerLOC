"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x10x10x10xf32>) -> tensor<?x10x10xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x10x10x10xf32>):
    %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %1 = "stablehlo.broadcast_in_dim"(%0) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<1xi32>
    %2 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %3 = "stablehlo.reshape"(%2) : (tensor<i32>) -> tensor<1xi32>
    %4 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %5 = "stablehlo.constant"() <{value = dense<10> : tensor<1xi32>}> : () -> tensor<1xi32>
    %6 = "stablehlo.constant"() <{value = dense<10> : tensor<1xi32>}> : () -> tensor<1xi32>
    %7 = "stablehlo.concatenate"(%3, %4, %5, %6) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %8 = "stablehlo.dynamic_gather"(%arg1, %1, %7) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [1], start_index_map = [1]>, indices_are_sorted = true}> : (tensor<?x10x10x10xf32>, tensor<1xi32>, tensor<4xi32>) -> tensor<?x10x10xf32>
    "func.return"(%8) : (tensor<?x10x10xf32>) -> ()
  }) : () -> ()
}) : () -> ()

