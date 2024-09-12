"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x4xf32>) -> tensor<4xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x4xf32>):
    %0 = "stablehlo.constant"() <{value = dense<-2> : tensor<i64>}> : () -> tensor<i64>
    %1 = "stablehlo.add"(%arg0, %0) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %2 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i64>
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %4 = "stablehlo.compare"(%1, %3) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %5 = "stablehlo.add"(%1, %2) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %6 = "stablehlo.select"(%4, %5, %1) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
    %7 = "stablehlo.broadcast_in_dim"(%6) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %8 = "stablehlo.gather"(%arg1, %7) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 4>}> : (tensor<?x4xf32>, tensor<1xi64>) -> tensor<4xf32>
    "func.return"(%8) : (tensor<4xf32>) -> ()
  }) : () -> ()
}) : () -> ()

