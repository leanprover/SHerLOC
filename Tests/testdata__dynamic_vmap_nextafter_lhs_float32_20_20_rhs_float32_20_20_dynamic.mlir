"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x20x20xf32>, tensor<?x20x20xf32>) -> tensor<?x20x20xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x20x20xf32>, %arg2: tensor<?x20x20xf32>):
    %0 = "stablehlo.constant"() <{value = dense<20> : tensor<1xi32>}> : () -> tensor<1xi32>
    %1 = "stablehlo.constant"() <{value = dense<-1> : tensor<i32>}> : () -> tensor<i32>
    %2 = "stablehlo.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %4 = "stablehlo.constant"() <{value = dense<2147483647> : tensor<i32>}> : () -> tensor<i32>
    %5 = "stablehlo.constant"() <{value = dense<-2147483648> : tensor<i32>}> : () -> tensor<i32>
    %6 = "stablehlo.constant"() <{value = dense<0x7FC00000> : tensor<f32>}> : () -> tensor<f32>
    %7 = "stablehlo.bitcast_convert"(%arg1) : (tensor<?x20x20xf32>) -> tensor<?x20x20xi32>
    %8 = "stablehlo.bitcast_convert"(%arg2) : (tensor<?x20x20xf32>) -> tensor<?x20x20xi32>
    %9 = "stablehlo.compare"(%arg1, %arg1) <{comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<?x20x20xf32>, tensor<?x20x20xf32>) -> tensor<?x20x20xi1>
    %10 = "stablehlo.compare"(%arg2, %arg2) <{comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<?x20x20xf32>, tensor<?x20x20xf32>) -> tensor<?x20x20xi1>
    %11 = "stablehlo.or"(%9, %10) : (tensor<?x20x20xi1>, tensor<?x20x20xi1>) -> tensor<?x20x20xi1>
    %12 = "stablehlo.get_dimension_size"(%arg1) <{dimension = 0 : i64}> : (tensor<?x20x20xf32>) -> tensor<i32>
    %13 = "stablehlo.reshape"(%12) : (tensor<i32>) -> tensor<1xi32>
    %14 = "stablehlo.concatenate"(%13, %0, %0) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %15 = "stablehlo.dynamic_broadcast_in_dim"(%6, %14) <{broadcast_dimensions = array<i64>}> : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %16 = "stablehlo.bitcast_convert"(%15) : (tensor<?x20x20xf32>) -> tensor<?x20x20xi32>
    %17 = "stablehlo.get_dimension_size"(%7) <{dimension = 0 : i64}> : (tensor<?x20x20xi32>) -> tensor<i32>
    %18 = "stablehlo.reshape"(%17) : (tensor<i32>) -> tensor<1xi32>
    %19 = "stablehlo.concatenate"(%18, %0, %0) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %20 = "stablehlo.dynamic_broadcast_in_dim"(%5, %19) <{broadcast_dimensions = array<i64>}> : (tensor<i32>, tensor<3xi32>) -> tensor<?x20x20xi32>
    %21 = "stablehlo.get_dimension_size"(%7) <{dimension = 0 : i64}> : (tensor<?x20x20xi32>) -> tensor<i32>
    %22 = "stablehlo.reshape"(%21) : (tensor<i32>) -> tensor<1xi32>
    %23 = "stablehlo.concatenate"(%22, %0, %0) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %24 = "stablehlo.dynamic_broadcast_in_dim"(%4, %23) <{broadcast_dimensions = array<i64>}> : (tensor<i32>, tensor<3xi32>) -> tensor<?x20x20xi32>
    %25 = "stablehlo.and"(%7, %24) : (tensor<?x20x20xi32>, tensor<?x20x20xi32>) -> tensor<?x20x20xi32>
    %26 = "stablehlo.and"(%8, %24) : (tensor<?x20x20xi32>, tensor<?x20x20xi32>) -> tensor<?x20x20xi32>
    %27 = "stablehlo.compare"(%arg1, %arg2) <{comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<?x20x20xf32>, tensor<?x20x20xf32>) -> tensor<?x20x20xi1>
    %28 = "stablehlo.get_dimension_size"(%7) <{dimension = 0 : i64}> : (tensor<?x20x20xi32>) -> tensor<i32>
    %29 = "stablehlo.reshape"(%28) : (tensor<i32>) -> tensor<1xi32>
    %30 = "stablehlo.concatenate"(%29, %0, %0) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %31 = "stablehlo.dynamic_broadcast_in_dim"(%3, %30) <{broadcast_dimensions = array<i64>}> : (tensor<i32>, tensor<3xi32>) -> tensor<?x20x20xi32>
    %32 = "stablehlo.compare"(%25, %31) <{comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<?x20x20xi32>, tensor<?x20x20xi32>) -> tensor<?x20x20xi1>
    %33 = "stablehlo.compare"(%26, %31) <{comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<?x20x20xi32>, tensor<?x20x20xi32>) -> tensor<?x20x20xi1>
    %34 = "stablehlo.and"(%7, %20) : (tensor<?x20x20xi32>, tensor<?x20x20xi32>) -> tensor<?x20x20xi32>
    %35 = "stablehlo.and"(%8, %20) : (tensor<?x20x20xi32>, tensor<?x20x20xi32>) -> tensor<?x20x20xi32>
    %36 = "stablehlo.get_dimension_size"(%7) <{dimension = 0 : i64}> : (tensor<?x20x20xi32>) -> tensor<i32>
    %37 = "stablehlo.reshape"(%36) : (tensor<i32>) -> tensor<1xi32>
    %38 = "stablehlo.concatenate"(%37, %0, %0) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %39 = "stablehlo.dynamic_broadcast_in_dim"(%2, %38) <{broadcast_dimensions = array<i64>}> : (tensor<i32>, tensor<3xi32>) -> tensor<?x20x20xi32>
    %40 = "stablehlo.or"(%35, %39) : (tensor<?x20x20xi32>, tensor<?x20x20xi32>) -> tensor<?x20x20xi32>
    %41 = "stablehlo.compare"(%34, %35) <{comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<?x20x20xi32>, tensor<?x20x20xi32>) -> tensor<?x20x20xi1>
    %42 = "stablehlo.compare"(%25, %26) <{comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<?x20x20xi32>, tensor<?x20x20xi32>) -> tensor<?x20x20xi1>
    %43 = "stablehlo.or"(%42, %41) : (tensor<?x20x20xi1>, tensor<?x20x20xi1>) -> tensor<?x20x20xi1>
    %44 = "stablehlo.get_dimension_size"(%7) <{dimension = 0 : i64}> : (tensor<?x20x20xi32>) -> tensor<i32>
    %45 = "stablehlo.reshape"(%44) : (tensor<i32>) -> tensor<1xi32>
    %46 = "stablehlo.concatenate"(%45, %0, %0) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %47 = "stablehlo.dynamic_broadcast_in_dim"(%1, %46) <{broadcast_dimensions = array<i64>}> : (tensor<i32>, tensor<3xi32>) -> tensor<?x20x20xi32>
    %48 = "stablehlo.select"(%43, %47, %39) : (tensor<?x20x20xi1>, tensor<?x20x20xi32>, tensor<?x20x20xi32>) -> tensor<?x20x20xi32>
    %49 = "stablehlo.add"(%7, %48) : (tensor<?x20x20xi32>, tensor<?x20x20xi32>) -> tensor<?x20x20xi32>
    %50 = "stablehlo.select"(%33, %8, %40) : (tensor<?x20x20xi1>, tensor<?x20x20xi32>, tensor<?x20x20xi32>) -> tensor<?x20x20xi32>
    %51 = "stablehlo.select"(%32, %50, %49) : (tensor<?x20x20xi1>, tensor<?x20x20xi32>, tensor<?x20x20xi32>) -> tensor<?x20x20xi32>
    %52 = "stablehlo.select"(%27, %8, %51) : (tensor<?x20x20xi1>, tensor<?x20x20xi32>, tensor<?x20x20xi32>) -> tensor<?x20x20xi32>
    %53 = "stablehlo.select"(%11, %16, %52) : (tensor<?x20x20xi1>, tensor<?x20x20xi32>, tensor<?x20x20xi32>) -> tensor<?x20x20xi32>
    %54 = "stablehlo.bitcast_convert"(%53) : (tensor<?x20x20xi32>) -> tensor<?x20x20xf32>
    "func.return"(%54) : (tensor<?x20x20xf32>) -> ()
  }) : () -> ()
}) : () -> ()

