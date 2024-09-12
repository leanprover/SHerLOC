"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x4x5xf32>, tensor<2xi32>) -> tensor<?x2x5xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg9: tensor<i64>, %arg10: tensor<?x4x5xf32>, %arg11: tensor<2xi32>):
    %77 = "func.call"(%arg9, %arg10, %arg11) <{callee = @_take}> : (tensor<i64>, tensor<?x4x5xf32>, tensor<2xi32>) -> tensor<?x2x5xf32>
    "func.return"(%77) : (tensor<?x2x5xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<?x4x5xf32>, tensor<2xi32>) -> tensor<?x2x5xf32>, sym_name = "_take", sym_visibility = "private"}> ({
  ^bb0(%arg4: tensor<i64>, %arg5: tensor<?x4x5xf32>, %arg6: tensor<2xi32>):
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %2 = "stablehlo.broadcast_in_dim"(%1) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<2xi32>
    %3 = "stablehlo.compare"(%arg6, %2) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
    %4 = "stablehlo.constant"() <{value = dense<4> : tensor<i32>}> : () -> tensor<i32>
    %5 = "stablehlo.broadcast_in_dim"(%4) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<2xi32>
    %6 = "stablehlo.add"(%arg6, %5) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
    %7 = "func.call"(%arg4, %3, %6, %arg6) <{callee = @_where}> : (tensor<i64>, tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
    %8 = "stablehlo.broadcast_in_dim"(%7) <{broadcast_dimensions = array<i64: 0>}> : (tensor<2xi32>) -> tensor<2x1xi32>
    %9 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi64>}> : () -> tensor<1xi64>
    %10 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi64>}> : () -> tensor<1xi64>
    %11 = "stablehlo.convert"(%arg4) : (tensor<i64>) -> tensor<i64>
    %12 = "stablehlo.broadcast_in_dim"(%11) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %13 = "stablehlo.constant"() <{value = dense<4> : tensor<i64>}> : () -> tensor<i64>
    %14 = "stablehlo.broadcast_in_dim"(%13) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %15 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
    %16 = "stablehlo.broadcast_in_dim"(%15) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %17 = "stablehlo.concatenate"(%12, %14, %16) <{dimension = 0 : i64}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
    %18 = "stablehlo.convert"(%8) : (tensor<2x1xi32>) -> tensor<2x1xi64>
    %19 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %20 = "stablehlo.broadcast_in_dim"(%19) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %21 = "stablehlo.compare"(%9, %20) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %22 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
    %23 = "stablehlo.broadcast_in_dim"(%22) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %24 = "stablehlo.add"(%9, %23) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %25 = "stablehlo.select"(%21, %24, %9) : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %26 = "stablehlo.convert"(%25) : (tensor<1xi64>) -> tensor<1xi32>
    %27 = "stablehlo.broadcast_in_dim"(%26) <{broadcast_dimensions = array<i64: 0>}> : (tensor<1xi32>) -> tensor<1x1xi32>
    %28 = "stablehlo.gather"(%17, %27) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<3xi64>, tensor<1x1xi32>) -> tensor<1xi64>
    %29 = "stablehlo.convert"(%arg4) : (tensor<i64>) -> tensor<i64>
    %30 = "stablehlo.broadcast_in_dim"(%29) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %31 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %32 = "stablehlo.broadcast_in_dim"(%31) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %33 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
    %34 = "stablehlo.broadcast_in_dim"(%33) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %35 = "stablehlo.concatenate"(%30, %32, %34) <{dimension = 0 : i64}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
    %36 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %37 = "stablehlo.broadcast_in_dim"(%36) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %38 = "stablehlo.compare"(%10, %37) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %39 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
    %40 = "stablehlo.broadcast_in_dim"(%39) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %41 = "stablehlo.add"(%10, %40) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %42 = "stablehlo.select"(%38, %41, %10) : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %43 = "stablehlo.convert"(%42) : (tensor<1xi64>) -> tensor<1xi32>
    %44 = "stablehlo.broadcast_in_dim"(%43) <{broadcast_dimensions = array<i64: 0>}> : (tensor<1xi32>) -> tensor<1x1xi32>
    %45 = "stablehlo.gather"(%35, %44) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<3xi64>, tensor<1x1xi32>) -> tensor<1xi64>
    %46 = "stablehlo.subtract"(%28, %45) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %47 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %48 = "stablehlo.broadcast_in_dim"(%47) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<2x1xi64>
    %49 = "stablehlo.compare"(%18, %48) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<2x1xi64>, tensor<2x1xi64>) -> tensor<2x1xi1>
    %50 = "stablehlo.broadcast_in_dim"(%46) <{broadcast_dimensions = array<i64: 1>}> : (tensor<1xi64>) -> tensor<1x1xi64>
    %51 = "stablehlo.broadcast_in_dim"(%50) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x1xi64>) -> tensor<2x1xi64>
    %52 = "stablehlo.compare"(%18, %51) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<2x1xi64>, tensor<2x1xi64>) -> tensor<2x1xi1>
    %53 = "stablehlo.and"(%49, %52) : (tensor<2x1xi1>, tensor<2x1xi1>) -> tensor<2x1xi1>
    %54 = "stablehlo.constant"() <{value = dense<true> : tensor<i1>}> : () -> tensor<i1>
    %55 = "stablehlo.reduce"(%53, %54) <{dimensions = array<i64: 1>}> ({
    ^bb0(%arg7: tensor<i1>, %arg8: tensor<i1>):
      %76 = "stablehlo.and"(%arg7, %arg8) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%76) : (tensor<i1>) -> ()
    }) : (tensor<2x1xi1>, tensor<i1>) -> tensor<2xi1>
    %56 = "stablehlo.convert"(%arg4) : (tensor<i64>) -> tensor<i32>
    %57 = "stablehlo.reshape"(%56) : (tensor<i32>) -> tensor<1xi32>
    %58 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %59 = "stablehlo.constant"() <{value = dense<5> : tensor<1xi32>}> : () -> tensor<1xi32>
    %60 = "stablehlo.concatenate"(%57, %58, %59) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %61 = "stablehlo.dynamic_gather"(%arg5, %18, %60) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>}> : (tensor<?x4x5xf32>, tensor<2x1xi64>, tensor<3xi32>) -> tensor<?x2x5xf32>
    %62 = "stablehlo.convert"(%arg4) : (tensor<i64>) -> tensor<i32>
    %63 = "stablehlo.reshape"(%62) : (tensor<i32>) -> tensor<1xi32>
    %64 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %65 = "stablehlo.constant"() <{value = dense<5> : tensor<1xi32>}> : () -> tensor<1xi32>
    %66 = "stablehlo.concatenate"(%63, %64, %65) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %67 = "stablehlo.dynamic_broadcast_in_dim"(%55, %66) <{broadcast_dimensions = array<i64: 1>}> : (tensor<2xi1>, tensor<3xi32>) -> tensor<?x2x5xi1>
    %68 = "stablehlo.constant"() <{value = dense<0x7FC00000> : tensor<f32>}> : () -> tensor<f32>
    %69 = "stablehlo.convert"(%arg4) : (tensor<i64>) -> tensor<i32>
    %70 = "stablehlo.reshape"(%69) : (tensor<i32>) -> tensor<1xi32>
    %71 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %72 = "stablehlo.constant"() <{value = dense<5> : tensor<1xi32>}> : () -> tensor<1xi32>
    %73 = "stablehlo.concatenate"(%70, %71, %72) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %74 = "stablehlo.dynamic_broadcast_in_dim"(%68, %73) <{broadcast_dimensions = array<i64>}> : (tensor<f32>, tensor<3xi32>) -> tensor<?x2x5xf32>
    %75 = "stablehlo.select"(%67, %61, %74) : (tensor<?x2x5xi1>, tensor<?x2x5xf32>, tensor<?x2x5xf32>) -> tensor<?x2x5xf32>
    "func.return"(%75) : (tensor<?x2x5xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>, sym_name = "_where", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<2xi1>, %arg2: tensor<2xi32>, %arg3: tensor<2xi32>):
    %0 = "stablehlo.select"(%arg1, %arg2, %arg3) : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
    "func.return"(%0) : (tensor<2xi32>) -> ()
  }) : () -> ()
}) : () -> ()

