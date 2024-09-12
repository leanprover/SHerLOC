"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x10x10x10xf32>, tensor<?xi32>) -> tensor<?x10x10xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg3: tensor<i64>, %arg4: tensor<?x10x10x10xf32>, %arg5: tensor<?xi32>):
    %12 = "func.call"(%arg3, %arg4, %arg5) <{callee = @_take}> : (tensor<i64>, tensor<?x10x10x10xf32>, tensor<?xi32>) -> tensor<?x10x10xf32>
    "func.return"(%12) : (tensor<?x10x10xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<?x10x10x10xf32>, tensor<?xi32>) -> tensor<?x10x10xf32>, sym_name = "_take", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x10x10x10xf32>, %arg2: tensor<?xi32>):
    %0 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %1 = "stablehlo.reshape"(%0) : (tensor<i32>) -> tensor<1xi32>
    %2 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = "stablehlo.concatenate"(%1, %2) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %4 = "stablehlo.dynamic_broadcast_in_dim"(%arg2, %3) <{broadcast_dimensions = array<i64: 0>}> : (tensor<?xi32>, tensor<2xi32>) -> tensor<?x1xi32>
    %5 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %6 = "stablehlo.reshape"(%5) : (tensor<i32>) -> tensor<1xi32>
    %7 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %8 = "stablehlo.concatenate"(%6, %7) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %9 = "stablehlo.dynamic_iota"(%8) <{iota_dimension = 0 : i64}> : (tensor<2xi32>) -> tensor<?x1xi32>
    %10 = "stablehlo.concatenate"(%9, %4) <{dimension = 1 : i64}> : (tensor<?x1xi32>, tensor<?x1xi32>) -> tensor<?x2xi32>
    %11 = "stablehlo.gather"(%arg1, %10) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1, 10, 10>}> : (tensor<?x10x10x10xf32>, tensor<?x2xi32>) -> tensor<?x10x10xf32>
    "func.return"(%11) : (tensor<?x10x10xf32>) -> ()
  }) : () -> ()
}) : () -> ()

