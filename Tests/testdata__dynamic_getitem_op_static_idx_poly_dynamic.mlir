"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<3x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<3x4xf32>, %arg2: tensor<?xi32>):
    %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %1 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %2 = "stablehlo.reshape"(%1) : (tensor<i32>) -> tensor<1xi32>
    %3 = "stablehlo.dynamic_broadcast_in_dim"(%0, %2) <{broadcast_dimensions = array<i64>}> : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
    %4 = "stablehlo.compare"(%arg2, %3) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi1>
    %5 = "stablehlo.constant"() <{value = dense<3> : tensor<i32>}> : () -> tensor<i32>
    %6 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %7 = "stablehlo.reshape"(%6) : (tensor<i32>) -> tensor<1xi32>
    %8 = "stablehlo.dynamic_broadcast_in_dim"(%5, %7) <{broadcast_dimensions = array<i64>}> : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
    %9 = "stablehlo.add"(%arg2, %8) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    %10 = "stablehlo.select"(%4, %9, %arg2) : (tensor<?xi1>, tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    %11 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %12 = "stablehlo.reshape"(%11) : (tensor<i32>) -> tensor<1xi32>
    %13 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %14 = "stablehlo.concatenate"(%12, %13) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %15 = "stablehlo.dynamic_broadcast_in_dim"(%10, %14) <{broadcast_dimensions = array<i64: 0>}> : (tensor<?xi32>, tensor<2xi32>) -> tensor<?x1xi32>
    %16 = "stablehlo.gather"(%arg1, %15) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 4>}> : (tensor<3x4xf32>, tensor<?x1xi32>) -> tensor<?x4xf32>
    "func.return"(%16) : (tensor<?x4xf32>) -> ()
  }) : () -> ()
}) : () -> ()

