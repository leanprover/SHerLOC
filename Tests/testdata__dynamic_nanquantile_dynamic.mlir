"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x5xf32>) -> tensor<f32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg15: tensor<i64>, %arg16: tensor<?x5xf32>):
    %100 = "stablehlo.constant"() <{value = dense<5.000000e-01> : tensor<f64>}> : () -> tensor<f64>
    %101 = "func.call"(%arg15, %arg16, %100) <{callee = @nanquantile}> : (tensor<i64>, tensor<?x5xf32>, tensor<f64>) -> tensor<f32>
    "func.return"(%101) : (tensor<f32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<?x5xf32>, tensor<f64>) -> tensor<f32>, sym_name = "nanquantile", sym_visibility = "private"}> ({
  ^bb0(%arg8: tensor<i64>, %arg9: tensor<?x5xf32>, %arg10: tensor<f64>):
    %9 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
    %10 = "stablehlo.multiply"(%arg8, %9) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %11 = "stablehlo.convert"(%10) : (tensor<i64>) -> tensor<i32>
    %12 = "stablehlo.reshape"(%11) : (tensor<i32>) -> tensor<1xi32>
    %13 = "stablehlo.dynamic_reshape"(%arg9, %12) : (tensor<?x5xf32>, tensor<1xi32>) -> tensor<?xf32>
    %14 = "func.call"(%arg8, %13) <{callee = @isnan}> : (tensor<i64>, tensor<?xf32>) -> tensor<?xi1>
    %15 = "stablehlo.constant"() <{value = dense<0x7FF8000000000000> : tensor<f64>}> : () -> tensor<f64>
    %16 = "func.call"(%arg8, %14, %15, %13) <{callee = @_where}> : (tensor<i64>, tensor<?xi1>, tensor<f64>, tensor<?xf32>) -> tensor<?xf32>
    %17 = "stablehlo.sort"(%16) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg13: tensor<f32>, %arg14: tensor<f32>):
      %69 = "stablehlo.bitcast_convert"(%arg13) : (tensor<f32>) -> tensor<i32>
      %70 = "stablehlo.bitcast_convert"(%arg13) : (tensor<f32>) -> tensor<ui32>
      %71 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %72 = "stablehlo.compare"(%arg13, %71) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %73 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
      %74 = "stablehlo.select"(%72, %73, %69) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %75 = "stablehlo.compare"(%arg13, %arg13) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %76 = "stablehlo.constant"() <{value = dense<2143289344> : tensor<i32>}> : () -> tensor<i32>
      %77 = "stablehlo.select"(%75, %76, %74) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %78 = "stablehlo.constant"() <{value = dense<2147483647> : tensor<ui32>}> : () -> tensor<ui32>
      %79 = "stablehlo.subtract"(%78, %70) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
      %80 = "stablehlo.bitcast_convert"(%79) : (tensor<ui32>) -> tensor<i32>
      %81 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
      %82 = "stablehlo.compare"(%77, %81) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %83 = "stablehlo.select"(%82, %80, %77) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %84 = "stablehlo.bitcast_convert"(%arg14) : (tensor<f32>) -> tensor<i32>
      %85 = "stablehlo.bitcast_convert"(%arg14) : (tensor<f32>) -> tensor<ui32>
      %86 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %87 = "stablehlo.compare"(%arg14, %86) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %88 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
      %89 = "stablehlo.select"(%87, %88, %84) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %90 = "stablehlo.compare"(%arg14, %arg14) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %91 = "stablehlo.constant"() <{value = dense<2143289344> : tensor<i32>}> : () -> tensor<i32>
      %92 = "stablehlo.select"(%90, %91, %89) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %93 = "stablehlo.constant"() <{value = dense<2147483647> : tensor<ui32>}> : () -> tensor<ui32>
      %94 = "stablehlo.subtract"(%93, %85) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
      %95 = "stablehlo.bitcast_convert"(%94) : (tensor<ui32>) -> tensor<i32>
      %96 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
      %97 = "stablehlo.compare"(%92, %96) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %98 = "stablehlo.select"(%97, %95, %92) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
      %99 = "stablehlo.compare"(%83, %98) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "stablehlo.return"(%99) : (tensor<i1>) -> ()
    }) : (tensor<?xf32>) -> tensor<?xf32>
    %18 = "func.call"(%arg8, %17) <{callee = @isnan_0}> : (tensor<i64>, tensor<?xf32>) -> tensor<?xi1>
    %19 = "stablehlo.not"(%18) : (tensor<?xi1>) -> tensor<?xi1>
    %20 = "stablehlo.convert"(%19) : (tensor<?xi1>) -> tensor<?xi32>
    %21 = "stablehlo.convert"(%20) : (tensor<?xi32>) -> tensor<?xf64>
    %22 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %23 = "stablehlo.reduce"(%21, %22) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg11: tensor<f64>, %arg12: tensor<f64>):
      %68 = "stablehlo.add"(%arg11, %arg12) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      "stablehlo.return"(%68) : (tensor<f64>) -> ()
    }) : (tensor<?xf64>, tensor<f64>) -> tensor<f64>
    %24 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %25 = "stablehlo.subtract"(%23, %24) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %26 = "stablehlo.multiply"(%arg10, %25) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %27 = "stablehlo.floor"(%26) : (tensor<f64>) -> tensor<f64>
    %28 = "stablehlo.ceil"(%26) : (tensor<f64>) -> tensor<f64>
    %29 = "stablehlo.subtract"(%26, %27) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %30 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %31 = "stablehlo.subtract"(%30, %29) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %32 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %33 = "stablehlo.subtract"(%23, %32) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %34 = "stablehlo.minimum"(%27, %33) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %35 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %36 = "stablehlo.maximum"(%35, %34) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %37 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %38 = "stablehlo.subtract"(%23, %37) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %39 = "stablehlo.minimum"(%28, %38) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %40 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %41 = "stablehlo.maximum"(%40, %39) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %42 = "stablehlo.convert"(%36) : (tensor<f64>) -> tensor<i64>
    %43 = "stablehlo.convert"(%41) : (tensor<f64>) -> tensor<i64>
    %44 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
    %45 = "stablehlo.multiply"(%arg8, %44) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %46 = "stablehlo.convert"(%45) : (tensor<i64>) -> tensor<i64>
    %47 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %48 = "stablehlo.compare"(%42, %47) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %49 = "stablehlo.add"(%42, %46) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %50 = "stablehlo.select"(%48, %49, %42) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
    %51 = "stablehlo.broadcast_in_dim"(%50) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %52 = "stablehlo.gather"(%17, %51) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = array<i64: 1>}> : (tensor<?xf32>, tensor<1xi64>) -> tensor<f32>
    %53 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
    %54 = "stablehlo.multiply"(%arg8, %53) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %55 = "stablehlo.convert"(%54) : (tensor<i64>) -> tensor<i64>
    %56 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %57 = "stablehlo.compare"(%43, %56) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %58 = "stablehlo.add"(%43, %55) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %59 = "stablehlo.select"(%57, %58, %43) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
    %60 = "stablehlo.broadcast_in_dim"(%59) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %61 = "stablehlo.gather"(%17, %60) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = array<i64: 1>}> : (tensor<?xf32>, tensor<1xi64>) -> tensor<f32>
    %62 = "stablehlo.convert"(%52) : (tensor<f32>) -> tensor<f64>
    %63 = "stablehlo.multiply"(%62, %31) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %64 = "stablehlo.convert"(%61) : (tensor<f32>) -> tensor<f64>
    %65 = "stablehlo.multiply"(%64, %29) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %66 = "stablehlo.add"(%63, %65) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    %67 = "stablehlo.convert"(%66) : (tensor<f64>) -> tensor<f32>
    "func.return"(%67) : (tensor<f32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<?xf32>) -> tensor<?xi1>, sym_name = "isnan", sym_visibility = "private"}> ({
  ^bb0(%arg6: tensor<i64>, %arg7: tensor<?xf32>):
    %8 = "stablehlo.compare"(%arg7, %arg7) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xi1>
    "func.return"(%8) : (tensor<?xi1>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<?xi1>, tensor<f64>, tensor<?xf32>) -> tensor<?xf32>, sym_name = "_where", sym_visibility = "private"}> ({
  ^bb0(%arg2: tensor<i64>, %arg3: tensor<?xi1>, %arg4: tensor<f64>, %arg5: tensor<?xf32>):
    %1 = "stablehlo.convert"(%arg4) : (tensor<f64>) -> tensor<f32>
    %2 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
    %3 = "stablehlo.multiply"(%arg2, %2) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %4 = "stablehlo.convert"(%3) : (tensor<i64>) -> tensor<i32>
    %5 = "stablehlo.reshape"(%4) : (tensor<i32>) -> tensor<1xi32>
    %6 = "stablehlo.dynamic_broadcast_in_dim"(%1, %5) <{broadcast_dimensions = array<i64>}> : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
    %7 = "stablehlo.select"(%arg3, %6, %arg5) : (tensor<?xi1>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    "func.return"(%7) : (tensor<?xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<?xf32>) -> tensor<?xi1>, sym_name = "isnan_0", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?xf32>):
    %0 = "stablehlo.compare"(%arg1, %arg1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xi1>
    "func.return"(%0) : (tensor<?xi1>) -> ()
  }) : () -> ()
}) : () -> ()

