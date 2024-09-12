"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x4xf32>) -> tensor<?x4xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg13: tensor<i64>, %arg14: tensor<?x4xf32>):
    %57 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
    %58 = "func.call"(%arg13, %arg14, %57) <{callee = @_roll}> : (tensor<i64>, tensor<?x4xf32>, tensor<i64>) -> tensor<?x4xf32>
    "func.return"(%58) : (tensor<?x4xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<?x4xf32>, tensor<i64>) -> tensor<?x4xf32>, sym_name = "_roll", sym_visibility = "private"}> ({
  ^bb0(%arg10: tensor<i64>, %arg11: tensor<?x4xf32>, %arg12: tensor<i64>):
    %46 = "stablehlo.constant"() <{value = dense<4> : tensor<i64>}> : () -> tensor<i64>
    %47 = "stablehlo.multiply"(%arg10, %46) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %48 = "stablehlo.convert"(%47) : (tensor<i64>) -> tensor<i32>
    %49 = "stablehlo.reshape"(%48) : (tensor<i32>) -> tensor<1xi32>
    %50 = "stablehlo.dynamic_reshape"(%arg11, %49) : (tensor<?x4xf32>, tensor<1xi32>) -> tensor<?xf32>
    %51 = "func.call"(%arg10, %50, %arg12) <{callee = @_roll_0}> : (tensor<i64>, tensor<?xf32>, tensor<i64>) -> tensor<?xf32>
    %52 = "stablehlo.convert"(%arg10) : (tensor<i64>) -> tensor<i32>
    %53 = "stablehlo.reshape"(%52) : (tensor<i32>) -> tensor<1xi32>
    %54 = "stablehlo.constant"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
    %55 = "stablehlo.concatenate"(%53, %54) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %56 = "stablehlo.dynamic_reshape"(%51, %55) : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x4xf32>
    "func.return"(%56) : (tensor<?x4xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<?xf32>, tensor<i64>) -> tensor<?xf32>, sym_name = "_roll_0", sym_visibility = "private"}> ({
  ^bb0(%arg7: tensor<i64>, %arg8: tensor<?xf32>, %arg9: tensor<i64>):
    %16 = "stablehlo.broadcast_in_dim"(%arg9) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %17 = "stablehlo.slice"(%16) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<1xi64>) -> tensor<1xi64>
    %18 = "stablehlo.reshape"(%17) : (tensor<1xi64>) -> tensor<i64>
    %19 = "stablehlo.constant"() <{value = dense<4> : tensor<i64>}> : () -> tensor<i64>
    %20 = "stablehlo.multiply"(%arg7, %19) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %21 = "stablehlo.convert"(%20) : (tensor<i64>) -> tensor<i32>
    %22 = "stablehlo.convert"(%18) : (tensor<i64>) -> tensor<i32>
    %23 = "stablehlo.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    %24 = "stablehlo.maximum"(%21, %23) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %25 = "func.call"(%arg7, %22, %24) <{callee = @remainder}> : (tensor<i64>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %26 = "stablehlo.concatenate"(%arg8, %arg8) <{dimension = 0 : i64}> : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    %27 = "stablehlo.subtract"(%21, %25) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %28 = "stablehlo.constant"() <{value = dense<8> : tensor<i64>}> : () -> tensor<i64>
    %29 = "stablehlo.multiply"(%arg7, %28) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %30 = "stablehlo.convert"(%29) : (tensor<i64>) -> tensor<i32>
    %31 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %32 = "stablehlo.compare"(%27, %31) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %33 = "stablehlo.add"(%27, %30) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %34 = "stablehlo.select"(%32, %33, %27) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %35 = "stablehlo.constant"() <{value = dense<4> : tensor<i64>}> : () -> tensor<i64>
    %36 = "stablehlo.multiply"(%arg7, %35) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %37 = "stablehlo.convert"(%34) : (tensor<i32>) -> tensor<i32>
    %38 = "stablehlo.reshape"(%37) : (tensor<i32>) -> tensor<1xi32>
    %39 = "stablehlo.convert"(%34) : (tensor<i32>) -> tensor<i32>
    %40 = "stablehlo.reshape"(%39) : (tensor<i32>) -> tensor<1xi32>
    %41 = "stablehlo.convert"(%36) : (tensor<i64>) -> tensor<i32>
    %42 = "stablehlo.reshape"(%41) : (tensor<i32>) -> tensor<1xi32>
    %43 = "stablehlo.add"(%40, %42) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %44 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %45 = "stablehlo.real_dynamic_slice"(%26, %38, %43, %44) : (tensor<?xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xf32>
    "func.return"(%45) : (tensor<?xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<i32>, tensor<i32>) -> tensor<i32>, sym_name = "remainder", sym_visibility = "private"}> ({
  ^bb0(%arg4: tensor<i64>, %arg5: tensor<i32>, %arg6: tensor<i32>):
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %2 = "stablehlo.compare"(%arg6, %1) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = "stablehlo.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    %4 = "func.call"(%arg4, %2, %3, %arg6) <{callee = @_where}> : (tensor<i64>, tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %5 = "stablehlo.remainder"(%arg5, %4) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %6 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %7 = "stablehlo.compare"(%5, %6) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %8 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %9 = "stablehlo.compare"(%5, %8) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %10 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %11 = "stablehlo.compare"(%4, %10) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %12 = "stablehlo.compare"(%9, %11) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %13 = "stablehlo.and"(%12, %7) : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %14 = "stablehlo.add"(%5, %4) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %15 = "stablehlo.select"(%13, %14, %5) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    "func.return"(%15) : (tensor<i32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>, sym_name = "_where", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<i1>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %0 = "stablehlo.select"(%arg1, %arg2, %arg3) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    "func.return"(%0) : (tensor<i32>) -> ()
  }) : () -> ()
}) : () -> ()

