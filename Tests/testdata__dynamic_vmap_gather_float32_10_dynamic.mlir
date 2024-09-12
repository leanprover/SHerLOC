"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x10xf32>, tensor<?xi32>) -> tensor<?xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg9: tensor<i64>, %arg10: tensor<?x10xf32>, %arg11: tensor<?xi32>):
    %81 = "func.call"(%arg9, %arg10, %arg11) <{callee = @_take}> : (tensor<i64>, tensor<?x10xf32>, tensor<?xi32>) -> tensor<?xf32>
    "func.return"(%81) : (tensor<?xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<?x10xf32>, tensor<?xi32>) -> tensor<?xf32>, sym_name = "_take", sym_visibility = "private"}> ({
  ^bb0(%arg4: tensor<i64>, %arg5: tensor<?x10xf32>, %arg6: tensor<?xi32>):
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
    %23 = "stablehlo.constant"() <{value = dense<[0, 1]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %24 = "stablehlo.constant"() <{value = dense<[0, 1]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %25 = "stablehlo.convert"(%arg4) : (tensor<i64>) -> tensor<i64>
    %26 = "stablehlo.broadcast_in_dim"(%25) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %27 = "stablehlo.constant"() <{value = dense<10> : tensor<i64>}> : () -> tensor<i64>
    %28 = "stablehlo.broadcast_in_dim"(%27) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %29 = "stablehlo.concatenate"(%26, %28) <{dimension = 0 : i64}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %30 = "stablehlo.convert"(%22) : (tensor<?x2xi32>) -> tensor<?x2xi64>
    %31 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %32 = "stablehlo.broadcast_in_dim"(%31) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<2xi64>
    %33 = "stablehlo.compare"(%23, %32) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi1>
    %34 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
    %35 = "stablehlo.broadcast_in_dim"(%34) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<2xi64>
    %36 = "stablehlo.add"(%23, %35) : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
    %37 = "stablehlo.select"(%33, %36, %23) : (tensor<2xi1>, tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
    %38 = "stablehlo.convert"(%37) : (tensor<2xi64>) -> tensor<2xi32>
    %39 = "stablehlo.broadcast_in_dim"(%38) <{broadcast_dimensions = array<i64: 0>}> : (tensor<2xi32>) -> tensor<2x1xi32>
    %40 = "stablehlo.gather"(%29, %39) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<2xi64>, tensor<2x1xi32>) -> tensor<2xi64>
    %41 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %42 = "stablehlo.broadcast_in_dim"(%41) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %43 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %44 = "stablehlo.broadcast_in_dim"(%43) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %45 = "stablehlo.concatenate"(%42, %44) <{dimension = 0 : i64}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %46 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %47 = "stablehlo.broadcast_in_dim"(%46) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<2xi64>
    %48 = "stablehlo.compare"(%24, %47) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi1>
    %49 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
    %50 = "stablehlo.broadcast_in_dim"(%49) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<2xi64>
    %51 = "stablehlo.add"(%24, %50) : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
    %52 = "stablehlo.select"(%48, %51, %24) : (tensor<2xi1>, tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
    %53 = "stablehlo.convert"(%52) : (tensor<2xi64>) -> tensor<2xi32>
    %54 = "stablehlo.broadcast_in_dim"(%53) <{broadcast_dimensions = array<i64: 0>}> : (tensor<2xi32>) -> tensor<2x1xi32>
    %55 = "stablehlo.gather"(%45, %54) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<2xi64>, tensor<2x1xi32>) -> tensor<2xi64>
    %56 = "stablehlo.subtract"(%40, %55) : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
    %57 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %58 = "stablehlo.convert"(%arg4) : (tensor<i64>) -> tensor<i32>
    %59 = "stablehlo.reshape"(%58) : (tensor<i32>) -> tensor<1xi32>
    %60 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %61 = "stablehlo.concatenate"(%59, %60) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %62 = "stablehlo.dynamic_broadcast_in_dim"(%57, %61) <{broadcast_dimensions = array<i64>}> : (tensor<i64>, tensor<2xi32>) -> tensor<?x2xi64>
    %63 = "stablehlo.compare"(%30, %62) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<?x2xi64>, tensor<?x2xi64>) -> tensor<?x2xi1>
    %64 = "stablehlo.broadcast_in_dim"(%56) <{broadcast_dimensions = array<i64: 1>}> : (tensor<2xi64>) -> tensor<1x2xi64>
    %65 = "stablehlo.convert"(%arg4) : (tensor<i64>) -> tensor<i32>
    %66 = "stablehlo.reshape"(%65) : (tensor<i32>) -> tensor<1xi32>
    %67 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %68 = "stablehlo.concatenate"(%66, %67) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %69 = "stablehlo.dynamic_broadcast_in_dim"(%64, %68) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x2xi64>, tensor<2xi32>) -> tensor<?x2xi64>
    %70 = "stablehlo.compare"(%30, %69) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<?x2xi64>, tensor<?x2xi64>) -> tensor<?x2xi1>
    %71 = "stablehlo.and"(%63, %70) : (tensor<?x2xi1>, tensor<?x2xi1>) -> tensor<?x2xi1>
    %72 = "stablehlo.constant"() <{value = dense<true> : tensor<i1>}> : () -> tensor<i1>
    %73 = "stablehlo.reduce"(%71, %72) <{dimensions = array<i64: 1>}> ({
    ^bb0(%arg7: tensor<i1>, %arg8: tensor<i1>):
      %80 = "stablehlo.and"(%arg7, %arg8) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%80) : (tensor<i1>) -> ()
    }) : (tensor<?x2xi1>, tensor<i1>) -> tensor<?xi1>
    %74 = "stablehlo.gather"(%arg5, %30) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1>}> : (tensor<?x10xf32>, tensor<?x2xi64>) -> tensor<?xf32>
    %75 = "stablehlo.constant"() <{value = dense<0x7FC00000> : tensor<f32>}> : () -> tensor<f32>
    %76 = "stablehlo.convert"(%arg4) : (tensor<i64>) -> tensor<i32>
    %77 = "stablehlo.reshape"(%76) : (tensor<i32>) -> tensor<1xi32>
    %78 = "stablehlo.dynamic_broadcast_in_dim"(%75, %77) <{broadcast_dimensions = array<i64>}> : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
    %79 = "stablehlo.select"(%73, %74, %78) : (tensor<?xi1>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    "func.return"(%79) : (tensor<?xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<?xi1>, tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>, sym_name = "_where", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?xi1>, %arg2: tensor<?xi32>, %arg3: tensor<?xi32>):
    %0 = "stablehlo.select"(%arg1, %arg2, %arg3) : (tensor<?xi1>, tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    "func.return"(%0) : (tensor<?xi32>) -> ()
  }) : () -> ()
}) : () -> ()

