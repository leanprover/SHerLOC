"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x10x10x10xf32>, tensor<?xi32>) -> tensor<?x10x10xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg9: tensor<i64>, %arg10: tensor<?x10x10x10xf32>, %arg11: tensor<?xi32>):
    %98 = "func.call"(%arg9, %arg10, %arg11) <{callee = @_take}> : (tensor<i64>, tensor<?x10x10x10xf32>, tensor<?xi32>) -> tensor<?x10x10xf32>
    "func.return"(%98) : (tensor<?x10x10xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<?x10x10x10xf32>, tensor<?xi32>) -> tensor<?x10x10xf32>, sym_name = "_take", sym_visibility = "private"}> ({
  ^bb0(%arg4: tensor<i64>, %arg5: tensor<?x10x10x10xf32>, %arg6: tensor<?xi32>):
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %2 = "stablehlo.convert"(%arg4) : (tensor<i64>) -> tensor<i32>
    %3 = "stablehlo.reshape"(%2) : (tensor<i32>) -> tensor<1xi32>
    %4 = "stablehlo.dynamic_broadcast_in_dim"(%1, %3) <{broadcast_dimensions = array<i64>}> : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
    %5 = "stablehlo.compare"(%arg6, %4) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi1>
    %6 = "stablehlo.constant"() <{value = dense<10> : tensor<i32>}> : () -> tensor<i32>
    %7 = "stablehlo.convert"(%arg4) : (tensor<i64>) -> tensor<i32>
    %8 = "stablehlo.reshape"(%7) : (tensor<i32>) -> tensor<1xi32>
    %9 = "stablehlo.dynamic_broadcast_in_dim"(%6, %8) <{broadcast_dimensions = array<i64>}> : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
    %10 = "stablehlo.add"(%arg6, %9) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    %11 = "func.call"(%arg4, %5, %10, %arg6) <{callee = @_where}> : (tensor<i64>, tensor<?xi1>, tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    %12 = "stablehlo.convert"(%arg4) : (tensor<i64>) -> tensor<i32>
    %13 = "stablehlo.reshape"(%12) : (tensor<i32>) -> tensor<1xi32>
    %14 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %15 = "stablehlo.concatenate"(%13, %14) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %16 = "stablehlo.dynamic_broadcast_in_dim"(%11, %15) <{broadcast_dimensions = array<i64: 0>}> : (tensor<?xi32>, tensor<2xi32>) -> tensor<?x1xi32>
    %17 = "stablehlo.convert"(%arg4) : (tensor<i64>) -> tensor<i32>
    %18 = "stablehlo.reshape"(%17) : (tensor<i32>) -> tensor<1xi32>
    %19 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %20 = "stablehlo.concatenate"(%18, %19) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %21 = "stablehlo.dynamic_iota"(%20) <{iota_dimension = 0 : i64}> : (tensor<2xi32>) -> tensor<?x1xi32>
    %22 = "stablehlo.concatenate"(%21, %16) <{dimension = 1 : i64}> : (tensor<?x1xi32>, tensor<?x1xi32>) -> tensor<?x2xi32>
    %23 = "stablehlo.constant"() <{value = dense<[0, 3]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %24 = "stablehlo.constant"() <{value = dense<[0, 3]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %25 = "stablehlo.convert"(%arg4) : (tensor<i64>) -> tensor<i64>
    %26 = "stablehlo.broadcast_in_dim"(%25) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %27 = "stablehlo.constant"() <{value = dense<10> : tensor<i64>}> : () -> tensor<i64>
    %28 = "stablehlo.broadcast_in_dim"(%27) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %29 = "stablehlo.constant"() <{value = dense<10> : tensor<i64>}> : () -> tensor<i64>
    %30 = "stablehlo.broadcast_in_dim"(%29) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %31 = "stablehlo.constant"() <{value = dense<10> : tensor<i64>}> : () -> tensor<i64>
    %32 = "stablehlo.broadcast_in_dim"(%31) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %33 = "stablehlo.concatenate"(%26, %28, %30, %32) <{dimension = 0 : i64}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
    %34 = "stablehlo.convert"(%22) : (tensor<?x2xi32>) -> tensor<?x2xi64>
    %35 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %36 = "stablehlo.broadcast_in_dim"(%35) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<2xi64>
    %37 = "stablehlo.compare"(%23, %36) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi1>
    %38 = "stablehlo.constant"() <{value = dense<4> : tensor<i64>}> : () -> tensor<i64>
    %39 = "stablehlo.broadcast_in_dim"(%38) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<2xi64>
    %40 = "stablehlo.add"(%23, %39) : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
    %41 = "stablehlo.select"(%37, %40, %23) : (tensor<2xi1>, tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
    %42 = "stablehlo.convert"(%41) : (tensor<2xi64>) -> tensor<2xi32>
    %43 = "stablehlo.broadcast_in_dim"(%42) <{broadcast_dimensions = array<i64: 0>}> : (tensor<2xi32>) -> tensor<2x1xi32>
    %44 = "stablehlo.gather"(%33, %43) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<4xi64>, tensor<2x1xi32>) -> tensor<2xi64>
    %45 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %46 = "stablehlo.broadcast_in_dim"(%45) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %47 = "stablehlo.constant"() <{value = dense<10> : tensor<i64>}> : () -> tensor<i64>
    %48 = "stablehlo.broadcast_in_dim"(%47) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %49 = "stablehlo.constant"() <{value = dense<10> : tensor<i64>}> : () -> tensor<i64>
    %50 = "stablehlo.broadcast_in_dim"(%49) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %51 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %52 = "stablehlo.broadcast_in_dim"(%51) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %53 = "stablehlo.concatenate"(%46, %48, %50, %52) <{dimension = 0 : i64}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
    %54 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %55 = "stablehlo.broadcast_in_dim"(%54) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<2xi64>
    %56 = "stablehlo.compare"(%24, %55) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi1>
    %57 = "stablehlo.constant"() <{value = dense<4> : tensor<i64>}> : () -> tensor<i64>
    %58 = "stablehlo.broadcast_in_dim"(%57) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<2xi64>
    %59 = "stablehlo.add"(%24, %58) : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
    %60 = "stablehlo.select"(%56, %59, %24) : (tensor<2xi1>, tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
    %61 = "stablehlo.convert"(%60) : (tensor<2xi64>) -> tensor<2xi32>
    %62 = "stablehlo.broadcast_in_dim"(%61) <{broadcast_dimensions = array<i64: 0>}> : (tensor<2xi32>) -> tensor<2x1xi32>
    %63 = "stablehlo.gather"(%53, %62) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<4xi64>, tensor<2x1xi32>) -> tensor<2xi64>
    %64 = "stablehlo.subtract"(%44, %63) : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
    %65 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %66 = "stablehlo.convert"(%arg4) : (tensor<i64>) -> tensor<i32>
    %67 = "stablehlo.reshape"(%66) : (tensor<i32>) -> tensor<1xi32>
    %68 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %69 = "stablehlo.concatenate"(%67, %68) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %70 = "stablehlo.dynamic_broadcast_in_dim"(%65, %69) <{broadcast_dimensions = array<i64>}> : (tensor<i64>, tensor<2xi32>) -> tensor<?x2xi64>
    %71 = "stablehlo.compare"(%34, %70) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<?x2xi64>, tensor<?x2xi64>) -> tensor<?x2xi1>
    %72 = "stablehlo.broadcast_in_dim"(%64) <{broadcast_dimensions = array<i64: 1>}> : (tensor<2xi64>) -> tensor<1x2xi64>
    %73 = "stablehlo.convert"(%arg4) : (tensor<i64>) -> tensor<i32>
    %74 = "stablehlo.reshape"(%73) : (tensor<i32>) -> tensor<1xi32>
    %75 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %76 = "stablehlo.concatenate"(%74, %75) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %77 = "stablehlo.dynamic_broadcast_in_dim"(%72, %76) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x2xi64>, tensor<2xi32>) -> tensor<?x2xi64>
    %78 = "stablehlo.compare"(%34, %77) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<?x2xi64>, tensor<?x2xi64>) -> tensor<?x2xi1>
    %79 = "stablehlo.and"(%71, %78) : (tensor<?x2xi1>, tensor<?x2xi1>) -> tensor<?x2xi1>
    %80 = "stablehlo.constant"() <{value = dense<true> : tensor<i1>}> : () -> tensor<i1>
    %81 = "stablehlo.reduce"(%79, %80) <{dimensions = array<i64: 1>}> ({
    ^bb0(%arg7: tensor<i1>, %arg8: tensor<i1>):
      %97 = "stablehlo.and"(%arg7, %arg8) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%97) : (tensor<i1>) -> ()
    }) : (tensor<?x2xi1>, tensor<i1>) -> tensor<?xi1>
    %82 = "stablehlo.gather"(%arg5, %34) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0, 3], start_index_map = [0, 3], index_vector_dim = 1>, slice_sizes = array<i64: 1, 10, 10, 1>}> : (tensor<?x10x10x10xf32>, tensor<?x2xi64>) -> tensor<?x10x10xf32>
    %83 = "stablehlo.convert"(%arg4) : (tensor<i64>) -> tensor<i32>
    %84 = "stablehlo.reshape"(%83) : (tensor<i32>) -> tensor<1xi32>
    %85 = "stablehlo.constant"() <{value = dense<10> : tensor<1xi32>}> : () -> tensor<1xi32>
    %86 = "stablehlo.constant"() <{value = dense<10> : tensor<1xi32>}> : () -> tensor<1xi32>
    %87 = "stablehlo.concatenate"(%84, %85, %86) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %88 = "stablehlo.dynamic_broadcast_in_dim"(%81, %87) <{broadcast_dimensions = array<i64: 0>}> : (tensor<?xi1>, tensor<3xi32>) -> tensor<?x10x10xi1>
    %89 = "stablehlo.constant"() <{value = dense<0x7FC00000> : tensor<f32>}> : () -> tensor<f32>
    %90 = "stablehlo.convert"(%arg4) : (tensor<i64>) -> tensor<i32>
    %91 = "stablehlo.reshape"(%90) : (tensor<i32>) -> tensor<1xi32>
    %92 = "stablehlo.constant"() <{value = dense<10> : tensor<1xi32>}> : () -> tensor<1xi32>
    %93 = "stablehlo.constant"() <{value = dense<10> : tensor<1xi32>}> : () -> tensor<1xi32>
    %94 = "stablehlo.concatenate"(%91, %92, %93) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %95 = "stablehlo.dynamic_broadcast_in_dim"(%89, %94) <{broadcast_dimensions = array<i64>}> : (tensor<f32>, tensor<3xi32>) -> tensor<?x10x10xf32>
    %96 = "stablehlo.select"(%88, %82, %95) : (tensor<?x10x10xi1>, tensor<?x10x10xf32>, tensor<?x10x10xf32>) -> tensor<?x10x10xf32>
    "func.return"(%96) : (tensor<?x10x10xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<?xi1>, tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>, sym_name = "_where", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?xi1>, %arg2: tensor<?xi32>, %arg3: tensor<?xi32>):
    %0 = "stablehlo.select"(%arg1, %arg2, %arg3) : (tensor<?xi1>, tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    "func.return"(%0) : (tensor<?xi32>) -> ()
  }) : () -> ()
}) : () -> ()

