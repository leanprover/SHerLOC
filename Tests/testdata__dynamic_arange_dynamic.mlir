"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?xf32>) -> tensor<?xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?xf32>):
    %0 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
    %1 = "stablehlo.multiply"(%arg0, %0) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %2 = "stablehlo.convert"(%1) : (tensor<i64>) -> tensor<i32>
    %3 = "stablehlo.reshape"(%2) : (tensor<i32>) -> tensor<1xi32>
    %4 = "stablehlo.dynamic_iota"(%3) <{iota_dimension = 0 : i64}> : (tensor<1xi32>) -> tensor<?xf32>
    %5 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %6 = "stablehlo.broadcast_in_dim"(%5) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %7 = "stablehlo.gather"(%arg1, %6) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = array<i64: 1>}> : (tensor<?xf32>, tensor<1xi64>) -> tensor<f32>
    %8 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
    %9 = "stablehlo.multiply"(%arg0, %8) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %10 = "stablehlo.convert"(%9) : (tensor<i64>) -> tensor<i32>
    %11 = "stablehlo.reshape"(%10) : (tensor<i32>) -> tensor<1xi32>
    %12 = "stablehlo.dynamic_broadcast_in_dim"(%7, %11) <{broadcast_dimensions = array<i64>}> : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
    %13 = "stablehlo.add"(%4, %12) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    "func.return"(%13) : (tensor<?xf32>) -> ()
  }) : () -> ()
}) : () -> ()

