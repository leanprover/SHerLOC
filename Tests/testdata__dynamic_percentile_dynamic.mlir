"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x5xf32>) -> tensor<f32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg16: tensor<i64>, %arg17: tensor<?x5xf32>):
    %93 = "stablehlo.constant"() <{value = dense<50> : tensor<i64>}> : () -> tensor<i64>
    %94 = "func.call"(%arg16, %arg17, %93) <{callee = @percentile}> : (tensor<i64>, tensor<?x5xf32>, tensor<i64>) -> tensor<f32>
    "func.return"(%94) : (tensor<f32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<?x5xf32>, tensor<i64>) -> tensor<f32>, sym_name = "percentile", sym_visibility = "private"}> ({
  ^bb0(%arg13: tensor<i64>, %arg14: tensor<?x5xf32>, %arg15: tensor<i64>):
    %89 = "stablehlo.convert"(%arg15) : (tensor<i64>) -> tensor<f64>
    %90 = "stablehlo.constant"() <{value = dense<1.000000e+02> : tensor<f64>}> : () -> tensor<f64>
    %91 = "stablehlo.divide"(%89, %90) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %92 = "func.call"(%arg13, %arg14, %91) <{callee = @quantile}> : (tensor<i64>, tensor<?x5xf32>, tensor<f64>) -> tensor<f32>
    "func.return"(%92) : (tensor<f32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<?x5xf32>, tensor<f64>) -> tensor<f32>, sym_name = "quantile", sym_visibility = "private"}> ({
  ^bb0(%arg6: tensor<i64>, %arg7: tensor<?x5xf32>, %arg8: tensor<f64>):
    %14 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
    %15 = "stablehlo.multiply"(%arg6, %14) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %16 = "stablehlo.convert"(%15) : (tensor<i64>) -> tensor<i32>
    %17 = "stablehlo.reshape"(%16) : (tensor<i32>) -> tensor<1xi32>
    %18 = "stablehlo.dynamic_reshape"(%arg7, %17) : (tensor<?x5xf32>, tensor<1xi32>) -> tensor<?xf32>
    %19 = "func.call"(%arg6, %18) <{callee = @isnan}> : (tensor<i64>, tensor<?xf32>) -> tensor<?xi1>
    %20 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
    %21 = "stablehlo.reduce"(%19, %20) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg11: tensor<i1>, %arg12: tensor<i1>):
      %88 = "stablehlo.or"(%arg11, %arg12) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%88) : (tensor<i1>) -> ()
    }) : (tensor<?xi1>, tensor<i1>) -> tensor<i1>
    %22 = "stablehlo.broadcast_in_dim"(%21) <{broadcast_dimensions = array<i64>}> : (tensor<i1>) -> tensor<1xi1>
    %23 = "stablehlo.constant"() <{value = dense<0x7FF8000000000000> : tensor<f64>}> : () -> tensor<f64>
    %24 = "func.call"(%arg6, %22, %23, %18) <{callee = @_where}> : (tensor<i64>, tensor<1xi1>, tensor<f64>, tensor<?xf32>) -> tensor<?xf32>
    %25 = "stablehlo.sort"(%24) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg9: tensor<f32>, %arg10: tensor<f32>):
      %57 = "stablehlo.bitcast_convert"(%arg9) : (tensor<f32>) -> tensor<i32>
      %58 = "stablehlo.bitcast_convert"(%arg9) : (tensor<f32>) -> tensor<ui32>
      %59 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %60 = "stablehlo.compare"(%arg9, %59) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %61 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
      %62 = "stablehlo.select"(%60, %61, %57) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %63 = "stablehlo.compare"(%arg9, %arg9) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %64 = "stablehlo.constant"() <{value = dense<2143289344> : tensor<i32>}> : () -> tensor<i32>
      %65 = "stablehlo.select"(%63, %64, %62) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %66 = "stablehlo.constant"() <{value = dense<2147483647> : tensor<ui32>}> : () -> tensor<ui32>
      %67 = "stablehlo.subtract"(%66, %58) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
      %68 = "stablehlo.bitcast_convert"(%67) : (tensor<ui32>) -> tensor<i32>
      %69 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
      %70 = "stablehlo.compare"(%65, %69) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %71 = "stablehlo.select"(%70, %68, %65) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %72 = "stablehlo.bitcast_convert"(%arg10) : (tensor<f32>) -> tensor<i32>
      %73 = "stablehlo.bitcast_convert"(%arg10) : (tensor<f32>) -> tensor<ui32>
      %74 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %75 = "stablehlo.compare"(%arg10, %74) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %76 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
      %77 = "stablehlo.select"(%75, %76, %72) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %78 = "stablehlo.compare"(%arg10, %arg10) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %79 = "stablehlo.constant"() <{value = dense<2143289344> : tensor<i32>}> : () -> tensor<i32>
      %80 = "stablehlo.select"(%78, %79, %77) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %81 = "stablehlo.constant"() <{value = dense<2147483647> : tensor<ui32>}> : () -> tensor<ui32>
      %82 = "stablehlo.subtract"(%81, %73) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
      %83 = "stablehlo.bitcast_convert"(%82) : (tensor<ui32>) -> tensor<i32>
      %84 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
      %85 = "stablehlo.compare"(%80, %84) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %86 = "stablehlo.select"(%85, %83, %80) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %87 = "stablehlo.compare"(%71, %86) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "stablehlo.return"(%87) : (tensor<i1>) -> ()
    }) : (tensor<?xf32>) -> tensor<?xf32>
    %26 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
    %27 = "stablehlo.multiply"(%arg6, %26) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %28 = "stablehlo.convert"(%27) : (tensor<i64>) -> tensor<f64>
    %29 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %30 = "stablehlo.subtract"(%28, %29) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %31 = "stablehlo.multiply"(%arg8, %30) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %32 = "stablehlo.floor"(%31) : (tensor<f64>) -> tensor<f64>
    %33 = "stablehlo.ceil"(%31) : (tensor<f64>) -> tensor<f64>
    %34 = "stablehlo.subtract"(%31, %32) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %35 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %36 = "stablehlo.subtract"(%35, %34) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %37 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %38 = "stablehlo.subtract"(%28, %37) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %39 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %40 = "stablehlo.clamp"(%39, %32, %38) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %41 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %42 = "stablehlo.subtract"(%28, %41) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %43 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %44 = "stablehlo.clamp"(%43, %33, %42) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %45 = "stablehlo.convert"(%40) : (tensor<f64>) -> tensor<i64>
    %46 = "stablehlo.convert"(%44) : (tensor<f64>) -> tensor<i64>
    %47 = "stablehlo.broadcast_in_dim"(%45) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %48 = "stablehlo.gather"(%25, %47) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, slice_sizes = array<i64: 1>}> : (tensor<?xf32>, tensor<1xi64>) -> tensor<f32>
    %49 = "stablehlo.broadcast_in_dim"(%46) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %50 = "stablehlo.gather"(%25, %49) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, slice_sizes = array<i64: 1>}> : (tensor<?xf32>, tensor<1xi64>) -> tensor<f32>
    %51 = "stablehlo.convert"(%48) : (tensor<f32>) -> tensor<f64>
    %52 = "stablehlo.multiply"(%51, %36) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %53 = "stablehlo.convert"(%50) : (tensor<f32>) -> tensor<f64>
    %54 = "stablehlo.multiply"(%53, %34) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %55 = "stablehlo.add"(%52, %54) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %56 = "stablehlo.convert"(%55) : (tensor<f64>) -> tensor<f32>
    "func.return"(%56) : (tensor<f32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<?xf32>) -> tensor<?xi1>, sym_name = "isnan", sym_visibility = "private"}> ({
  ^bb0(%arg4: tensor<i64>, %arg5: tensor<?xf32>):
    %13 = "stablehlo.compare"(%arg5, %arg5) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xi1>
    "func.return"(%13) : (tensor<?xi1>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<1xi1>, tensor<f64>, tensor<?xf32>) -> tensor<?xf32>, sym_name = "_where", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<1xi1>, %arg2: tensor<f64>, %arg3: tensor<?xf32>):
    %0 = "stablehlo.convert"(%arg2) : (tensor<f64>) -> tensor<f32>
    %1 = "stablehlo.reshape"(%arg1) : (tensor<1xi1>) -> tensor<i1>
    %2 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
    %3 = "stablehlo.multiply"(%arg0, %2) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %4 = "stablehlo.convert"(%3) : (tensor<i64>) -> tensor<i32>
    %5 = "stablehlo.reshape"(%4) : (tensor<i32>) -> tensor<1xi32>
    %6 = "stablehlo.dynamic_broadcast_in_dim"(%1, %5) <{broadcast_dimensions = array<i64>}> : (tensor<i1>, tensor<1xi32>) -> tensor<?xi1>
    %7 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
    %8 = "stablehlo.multiply"(%arg0, %7) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %9 = "stablehlo.convert"(%8) : (tensor<i64>) -> tensor<i32>
    %10 = "stablehlo.reshape"(%9) : (tensor<i32>) -> tensor<1xi32>
    %11 = "stablehlo.dynamic_broadcast_in_dim"(%0, %10) <{broadcast_dimensions = array<i64>}> : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
    %12 = "stablehlo.select"(%6, %11, %arg3) : (tensor<?xi1>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    "func.return"(%12) : (tensor<?xf32>) -> ()
  }) : () -> ()
}) : () -> ()

