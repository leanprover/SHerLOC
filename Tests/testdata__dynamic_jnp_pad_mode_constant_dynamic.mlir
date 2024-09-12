"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x5xf32>) -> tensor<?x11xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg3: tensor<i64>, %arg4: tensor<?x5xf32>):
    %40 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %41 = "func.call"(%arg3, %arg4, %40) <{callee = @_pad}> : (tensor<i64>, tensor<?x5xf32>, tensor<i64>) -> tensor<?x11xf32>
    "func.return"(%41) : (tensor<?x11xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<?x5xf32>, tensor<i64>) -> tensor<?x11xf32>, sym_name = "_pad", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x5xf32>, %arg2: tensor<i64>):
    %0 = "stablehlo.broadcast_in_dim"(%arg2) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<2x2xi64>
    %1 = "stablehlo.convert"(%0) : (tensor<2x2xi64>) -> tensor<2x2xf32>
    %2 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %3 = "stablehlo.broadcast_in_dim"(%2) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<1xi32>
    %4 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %5 = "stablehlo.broadcast_in_dim"(%4) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<1xi32>
    %6 = "stablehlo.concatenate"(%3, %5) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %7 = "stablehlo.gather"(%1, %6) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<2x2xf32>, tensor<2xi32>) -> tensor<f32>
    %8 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %9 = "stablehlo.reshape"(%8) : (tensor<i32>) -> tensor<1xi32>
    %10 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %11 = "stablehlo.concatenate"(%9, %10) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %12 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %13 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %14 = "stablehlo.concatenate"(%12, %13) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %15 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %16 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %17 = "stablehlo.concatenate"(%15, %16) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %18 = "stablehlo.dynamic_pad"(%arg1, %7, %11, %14, %17) : (tensor<?x5xf32>, tensor<f32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x5xf32>
    %19 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %20 = "stablehlo.broadcast_in_dim"(%19) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<1xi32>
    %21 = "stablehlo.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    %22 = "stablehlo.broadcast_in_dim"(%21) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<1xi32>
    %23 = "stablehlo.concatenate"(%20, %22) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %24 = "stablehlo.gather"(%1, %23) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<2x2xf32>, tensor<2xi32>) -> tensor<f32>
    %25 = "stablehlo.pad"(%18, %24) <{edge_padding_high = array<i64: 0, 0>, edge_padding_low = array<i64: 0, 0>, interior_padding = array<i64: 0, 0>}> : (tensor<?x5xf32>, tensor<f32>) -> tensor<?x5xf32>
    %26 = "stablehlo.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    %27 = "stablehlo.broadcast_in_dim"(%26) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<1xi32>
    %28 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %29 = "stablehlo.broadcast_in_dim"(%28) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<1xi32>
    %30 = "stablehlo.concatenate"(%27, %29) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %31 = "stablehlo.gather"(%1, %30) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<2x2xf32>, tensor<2xi32>) -> tensor<f32>
    %32 = "stablehlo.pad"(%25, %31) <{edge_padding_high = array<i64: 0, 0>, edge_padding_low = array<i64: 0, 5>, interior_padding = array<i64: 0, 0>}> : (tensor<?x5xf32>, tensor<f32>) -> tensor<?x10xf32>
    %33 = "stablehlo.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    %34 = "stablehlo.broadcast_in_dim"(%33) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<1xi32>
    %35 = "stablehlo.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    %36 = "stablehlo.broadcast_in_dim"(%35) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<1xi32>
    %37 = "stablehlo.concatenate"(%34, %36) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %38 = "stablehlo.gather"(%1, %37) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1]>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<2x2xf32>, tensor<2xi32>) -> tensor<f32>
    %39 = "stablehlo.pad"(%32, %38) <{edge_padding_high = array<i64: 0, 1>, edge_padding_low = array<i64: 0, 0>, interior_padding = array<i64: 0, 0>}> : (tensor<?x10xf32>, tensor<f32>) -> tensor<?x11xf32>
    "func.return"(%39) : (tensor<?x11xf32>) -> ()
  }) : () -> ()
}) : () -> ()

