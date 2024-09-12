"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x2xf32>, tensor<?x1xi32>) -> tensor<?x1xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg5: tensor<i64>, %arg6: tensor<?x2xf32>, %arg7: tensor<?x1xi32>):
    %90 = "func.call"(%arg5, %arg6, %arg7) <{callee = @take_along_axis}> : (tensor<i64>, tensor<?x2xf32>, tensor<?x1xi32>) -> tensor<?x1xf32>
    "func.return"(%90) : (tensor<?x1xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<?x2xf32>, tensor<?x1xi32>) -> tensor<?x1xf32>, sym_name = "take_along_axis", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x2xf32>, %arg2: tensor<?x1xi32>):
    %0 = "stablehlo.convert"(%arg2) : (tensor<?x1xi32>) -> tensor<?x1xi64>
    %1 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %2 = "stablehlo.reshape"(%1) : (tensor<i32>) -> tensor<1xi32>
    %3 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %5 = "stablehlo.concatenate"(%2, %3, %4) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %6 = "stablehlo.dynamic_iota"(%5) <{iota_dimension = 0 : i64}> : (tensor<3xi32>) -> tensor<?x1x1xi64>
    %7 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %8 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %9 = "stablehlo.reshape"(%8) : (tensor<i32>) -> tensor<1xi32>
    %10 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %11 = "stablehlo.concatenate"(%9, %10) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %12 = "stablehlo.dynamic_broadcast_in_dim"(%7, %11) <{broadcast_dimensions = array<i64>}> : (tensor<i64>, tensor<2xi32>) -> tensor<?x1xi64>
    %13 = "stablehlo.compare"(%0, %12) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<?x1xi64>, tensor<?x1xi64>) -> tensor<?x1xi1>
    %14 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
    %15 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %16 = "stablehlo.reshape"(%15) : (tensor<i32>) -> tensor<1xi32>
    %17 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %18 = "stablehlo.concatenate"(%16, %17) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %19 = "stablehlo.dynamic_broadcast_in_dim"(%14, %18) <{broadcast_dimensions = array<i64>}> : (tensor<i64>, tensor<2xi32>) -> tensor<?x1xi64>
    %20 = "stablehlo.add"(%0, %19) : (tensor<?x1xi64>, tensor<?x1xi64>) -> tensor<?x1xi64>
    %21 = "stablehlo.select"(%13, %20, %0) : (tensor<?x1xi1>, tensor<?x1xi64>, tensor<?x1xi64>) -> tensor<?x1xi64>
    %22 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %23 = "stablehlo.reshape"(%22) : (tensor<i32>) -> tensor<1xi32>
    %24 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %25 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %26 = "stablehlo.concatenate"(%23, %24, %25) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %27 = "stablehlo.dynamic_reshape"(%21, %26) : (tensor<?x1xi64>, tensor<3xi32>) -> tensor<?x1x1xi64>
    %28 = "stablehlo.concatenate"(%6, %27) <{dimension = 2 : i64}> : (tensor<?x1x1xi64>, tensor<?x1x1xi64>) -> tensor<?x1x2xi64>
    %29 = "stablehlo.constant"() <{value = dense<[0, 1]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %30 = "stablehlo.constant"() <{value = dense<[0, 1]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %31 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i64>
    %32 = "stablehlo.broadcast_in_dim"(%31) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %33 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
    %34 = "stablehlo.broadcast_in_dim"(%33) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %35 = "stablehlo.concatenate"(%32, %34) <{dimension = 0 : i64}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %36 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %37 = "stablehlo.broadcast_in_dim"(%36) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<2xi64>
    %38 = "stablehlo.compare"(%29, %37) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi1>
    %39 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
    %40 = "stablehlo.broadcast_in_dim"(%39) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<2xi64>
    %41 = "stablehlo.add"(%29, %40) : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
    %42 = "stablehlo.select"(%38, %41, %29) : (tensor<2xi1>, tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
    %43 = "stablehlo.convert"(%42) : (tensor<2xi64>) -> tensor<2xi32>
    %44 = "stablehlo.broadcast_in_dim"(%43) <{broadcast_dimensions = array<i64: 0>}> : (tensor<2xi32>) -> tensor<2x1xi32>
    %45 = "stablehlo.gather"(%35, %44) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<2xi64>, tensor<2x1xi32>) -> tensor<2xi64>
    %46 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %47 = "stablehlo.broadcast_in_dim"(%46) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %48 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %49 = "stablehlo.broadcast_in_dim"(%48) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %50 = "stablehlo.concatenate"(%47, %49) <{dimension = 0 : i64}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %51 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %52 = "stablehlo.broadcast_in_dim"(%51) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<2xi64>
    %53 = "stablehlo.compare"(%30, %52) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi1>
    %54 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
    %55 = "stablehlo.broadcast_in_dim"(%54) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<2xi64>
    %56 = "stablehlo.add"(%30, %55) : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
    %57 = "stablehlo.select"(%53, %56, %30) : (tensor<2xi1>, tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
    %58 = "stablehlo.convert"(%57) : (tensor<2xi64>) -> tensor<2xi32>
    %59 = "stablehlo.broadcast_in_dim"(%58) <{broadcast_dimensions = array<i64: 0>}> : (tensor<2xi32>) -> tensor<2x1xi32>
    %60 = "stablehlo.gather"(%50, %59) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<2xi64>, tensor<2x1xi32>) -> tensor<2xi64>
    %61 = "stablehlo.subtract"(%45, %60) : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
    %62 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %63 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %64 = "stablehlo.reshape"(%63) : (tensor<i32>) -> tensor<1xi32>
    %65 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %66 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %67 = "stablehlo.concatenate"(%64, %65, %66) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %68 = "stablehlo.dynamic_broadcast_in_dim"(%62, %67) <{broadcast_dimensions = array<i64>}> : (tensor<i64>, tensor<3xi32>) -> tensor<?x1x2xi64>
    %69 = "stablehlo.compare"(%28, %68) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<?x1x2xi64>, tensor<?x1x2xi64>) -> tensor<?x1x2xi1>
    %70 = "stablehlo.broadcast_in_dim"(%61) <{broadcast_dimensions = array<i64: 2>}> : (tensor<2xi64>) -> tensor<1x1x2xi64>
    %71 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %72 = "stablehlo.reshape"(%71) : (tensor<i32>) -> tensor<1xi32>
    %73 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %74 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %75 = "stablehlo.concatenate"(%72, %73, %74) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %76 = "stablehlo.dynamic_broadcast_in_dim"(%70, %75) <{broadcast_dimensions = array<i64: 0, 1, 2>}> : (tensor<1x1x2xi64>, tensor<3xi32>) -> tensor<?x1x2xi64>
    %77 = "stablehlo.compare"(%28, %76) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<?x1x2xi64>, tensor<?x1x2xi64>) -> tensor<?x1x2xi1>
    %78 = "stablehlo.and"(%69, %77) : (tensor<?x1x2xi1>, tensor<?x1x2xi1>) -> tensor<?x1x2xi1>
    %79 = "stablehlo.constant"() <{value = dense<true> : tensor<i1>}> : () -> tensor<i1>
    %80 = "stablehlo.reduce"(%78, %79) <{dimensions = array<i64: 2>}> ({
    ^bb0(%arg3: tensor<i1>, %arg4: tensor<i1>):
      %89 = "stablehlo.and"(%arg3, %arg4) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%89) : (tensor<i1>) -> ()
    }) : (tensor<?x1x2xi1>, tensor<i1>) -> tensor<?x1xi1>
    %81 = "stablehlo.gather"(%arg1, %28) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1>}> : (tensor<?x2xf32>, tensor<?x1x2xi64>) -> tensor<?x1xf32>
    %82 = "stablehlo.constant"() <{value = dense<0x7FC00000> : tensor<f32>}> : () -> tensor<f32>
    %83 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %84 = "stablehlo.reshape"(%83) : (tensor<i32>) -> tensor<1xi32>
    %85 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %86 = "stablehlo.concatenate"(%84, %85) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %87 = "stablehlo.dynamic_broadcast_in_dim"(%82, %86) <{broadcast_dimensions = array<i64>}> : (tensor<f32>, tensor<2xi32>) -> tensor<?x1xf32>
    %88 = "stablehlo.select"(%80, %81, %87) : (tensor<?x1xi1>, tensor<?x1xf32>, tensor<?x1xf32>) -> tensor<?x1xf32>
    "func.return"(%88) : (tensor<?x1xf32>) -> ()
  }) : () -> ()
}) : () -> ()

