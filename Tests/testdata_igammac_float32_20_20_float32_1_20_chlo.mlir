"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %5:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<20x20xf32>, tensor<1x20xf32>)
    %6 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xf32>
    %7 = "stablehlo.broadcast_in_dim"(%5#1) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x20xf32>) -> tensor<20x20xf32>
    %8 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %9 = "stablehlo.broadcast_in_dim"(%8) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %10 = "stablehlo.compare"(%7, %9) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %11 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %12 = "stablehlo.broadcast_in_dim"(%11) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %13 = "stablehlo.compare"(%5#0, %12) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %14 = "stablehlo.or"(%10, %13) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %15 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %16 = "stablehlo.broadcast_in_dim"(%15) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %17 = "stablehlo.compare"(%7, %16) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %18 = "stablehlo.compare"(%7, %5#0) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %19 = "stablehlo.or"(%17, %18) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %20 = "stablehlo.log"(%7) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %21 = "stablehlo.multiply"(%5#0, %20) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %22 = "stablehlo.subtract"(%21, %7) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %23 = "chlo.lgamma"(%5#0) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %24 = "stablehlo.subtract"(%22, %23) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %25 = "stablehlo.constant"() <{value = dense<3.40282347E+38> : tensor<f32>}> : () -> tensor<f32>
    %26 = "stablehlo.log"(%25) : (tensor<f32>) -> tensor<f32>
    %27 = "stablehlo.negate"(%26) : (tensor<f32>) -> tensor<f32>
    %28 = "stablehlo.broadcast_in_dim"(%27) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %29 = "stablehlo.compare"(%24, %28) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %30 = "stablehlo.or"(%14, %29) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %31 = "stablehlo.not"(%30) : (tensor<20x20xi1>) -> tensor<20x20xi1>
    %32 = "stablehlo.exponential"(%24) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %33 = "stablehlo.and"(%31, %19) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %34 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %35 = "stablehlo.broadcast_in_dim"(%34) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %36 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %37 = "stablehlo.broadcast_in_dim"(%36) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %38 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %39 = "stablehlo.broadcast_in_dim"(%38) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %40 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %41 = "stablehlo.broadcast_in_dim"(%40) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %42:7 = "stablehlo.while"(%33, %5#0, %35, %37, %7, %39, %41) ({
    ^bb0(%arg40: tensor<20x20xi1>, %arg41: tensor<20x20xf32>, %arg42: tensor<20x20xf32>, %arg43: tensor<20x20xf32>, %arg44: tensor<20x20xf32>, %arg45: tensor<20x20xf32>, %arg46: tensor<20x20xf32>):
      %223 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
      %224 = "stablehlo.reduce"(%arg40, %223) <{dimensions = array<i64: 0, 1>}> ({
      ^bb0(%arg47: tensor<i1>, %arg48: tensor<i1>):
        %225 = "stablehlo.or"(%arg47, %arg48) : (tensor<i1>, tensor<i1>) -> tensor<i1>
        "stablehlo.return"(%225) : (tensor<i1>) -> ()
      }) : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%224) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg33: tensor<20x20xi1>, %arg34: tensor<20x20xf32>, %arg35: tensor<20x20xf32>, %arg36: tensor<20x20xf32>, %arg37: tensor<20x20xf32>, %arg38: tensor<20x20xf32>, %arg39: tensor<20x20xf32>):
      %199 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %200 = "stablehlo.broadcast_in_dim"(%199) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %201 = "stablehlo.add"(%arg34, %200) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %202 = "stablehlo.divide"(%arg37, %201) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %203 = "stablehlo.multiply"(%arg38, %202) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %204 = "stablehlo.multiply"(%arg35, %arg37) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %205 = "stablehlo.multiply"(%201, %201) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %206 = "stablehlo.divide"(%204, %205) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %207 = "stablehlo.subtract"(%203, %206) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %208 = "stablehlo.add"(%arg39, %207) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %209 = "stablehlo.divide"(%arg37, %201) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %210 = "stablehlo.multiply"(%arg35, %209) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %211 = "stablehlo.add"(%arg36, %210) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %212 = "stablehlo.divide"(%210, %211) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %213 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %214 = "stablehlo.broadcast_in_dim"(%213) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %215 = "stablehlo.compare"(%212, %214) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %216 = "stablehlo.and"(%arg33, %215) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
      %217 = "stablehlo.select"(%arg33, %201, %arg34) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %218 = "stablehlo.select"(%arg33, %210, %arg35) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %219 = "stablehlo.select"(%arg33, %211, %arg36) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %220 = "stablehlo.select"(%arg33, %arg37, %arg37) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %221 = "stablehlo.select"(%arg33, %207, %arg38) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %222 = "stablehlo.select"(%arg33, %208, %arg39) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      "stablehlo.return"(%216, %217, %218, %219, %220, %221, %222) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    }) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>) -> (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>)
    %43 = "stablehlo.multiply"(%42#3, %32) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %44 = "stablehlo.divide"(%43, %5#0) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %45 = "stablehlo.not"(%19) : (tensor<20x20xi1>) -> tensor<20x20xi1>
    %46 = "stablehlo.and"(%31, %45) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %47 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %48 = "stablehlo.broadcast_in_dim"(%47) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %49 = "stablehlo.subtract"(%48, %5#0) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %50 = "stablehlo.add"(%7, %49) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %51 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %52 = "stablehlo.broadcast_in_dim"(%51) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %53 = "stablehlo.add"(%50, %52) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %54 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %55 = "stablehlo.broadcast_in_dim"(%54) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %56 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %57 = "stablehlo.broadcast_in_dim"(%56) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %58 = "stablehlo.add"(%7, %57) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %59 = "stablehlo.multiply"(%53, %7) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %60 = "stablehlo.divide"(%58, %59) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %61 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %62 = "stablehlo.broadcast_in_dim"(%61) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %63 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %64 = "stablehlo.broadcast_in_dim"(%63) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %65 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %66 = "stablehlo.broadcast_in_dim"(%65) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %67 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %68 = "stablehlo.broadcast_in_dim"(%67) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %69 = "stablehlo.negate"(%7) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %70 = "stablehlo.multiply"(%60, %69) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %71 = "stablehlo.subtract"(%68, %70) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %72 = "stablehlo.divide"(%71, %59) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %73 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %74:15 = "stablehlo.while"(%46, %60, %62, %49, %53, %73, %58, %59, %55, %7, %64, %66, %68, %69, %72) ({
    ^bb0(%arg16: tensor<20x20xi1>, %arg17: tensor<20x20xf32>, %arg18: tensor<20x20xf32>, %arg19: tensor<20x20xf32>, %arg20: tensor<20x20xf32>, %arg21: tensor<f32>, %arg22: tensor<20x20xf32>, %arg23: tensor<20x20xf32>, %arg24: tensor<20x20xf32>, %arg25: tensor<20x20xf32>, %arg26: tensor<20x20xf32>, %arg27: tensor<20x20xf32>, %arg28: tensor<20x20xf32>, %arg29: tensor<20x20xf32>, %arg30: tensor<20x20xf32>):
      %193 = "stablehlo.constant"() <{value = dense<2.000000e+03> : tensor<f32>}> : () -> tensor<f32>
      %194 = "stablehlo.compare"(%arg21, %193) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %195 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
      %196 = "stablehlo.reduce"(%arg16, %195) <{dimensions = array<i64: 0, 1>}> ({
      ^bb0(%arg31: tensor<i1>, %arg32: tensor<i1>):
        %198 = "stablehlo.or"(%arg31, %arg32) : (tensor<i1>, tensor<i1>) -> tensor<i1>
        "stablehlo.return"(%198) : (tensor<i1>) -> ()
      }) : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      %197 = "stablehlo.and"(%194, %196) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%197) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg1: tensor<20x20xi1>, %arg2: tensor<20x20xf32>, %arg3: tensor<20x20xf32>, %arg4: tensor<20x20xf32>, %arg5: tensor<20x20xf32>, %arg6: tensor<f32>, %arg7: tensor<20x20xf32>, %arg8: tensor<20x20xf32>, %arg9: tensor<20x20xf32>, %arg10: tensor<20x20xf32>, %arg11: tensor<20x20xf32>, %arg12: tensor<20x20xf32>, %arg13: tensor<20x20xf32>, %arg14: tensor<20x20xf32>, %arg15: tensor<20x20xf32>):
      %89 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %90 = "stablehlo.add"(%arg6, %89) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      %91 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %92 = "stablehlo.broadcast_in_dim"(%91) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %93 = "stablehlo.add"(%arg4, %92) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %94 = "stablehlo.constant"() <{value = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %95 = "stablehlo.broadcast_in_dim"(%94) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %96 = "stablehlo.add"(%arg5, %95) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %97 = "stablehlo.broadcast_in_dim"(%90) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %98 = "stablehlo.multiply"(%93, %97) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %99 = "stablehlo.multiply"(%arg7, %96) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %100 = "stablehlo.multiply"(%arg9, %98) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %101 = "stablehlo.subtract"(%99, %100) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %102 = "stablehlo.multiply"(%arg8, %96) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %103 = "stablehlo.multiply"(%arg10, %98) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %104 = "stablehlo.subtract"(%102, %103) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %105 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %106 = "stablehlo.broadcast_in_dim"(%105) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %107 = "stablehlo.compare"(%104, %106) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %108 = "stablehlo.divide"(%101, %104) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %109 = "stablehlo.subtract"(%arg2, %108) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %110 = "stablehlo.divide"(%109, %108) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %111 = "stablehlo.abs"(%110) : (tensor<20x20xf32>) -> tensor<20x20xf32>
      %112 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %113 = "stablehlo.broadcast_in_dim"(%112) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %114 = "stablehlo.select"(%107, %111, %113) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %115 = "stablehlo.select"(%107, %108, %arg2) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %116 = "stablehlo.multiply"(%arg13, %96) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %117 = "stablehlo.subtract"(%116, %arg7) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %118 = "stablehlo.multiply"(%arg11, %98) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %119 = "stablehlo.subtract"(%117, %118) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %120 = "stablehlo.broadcast_in_dim"(%90) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %121 = "stablehlo.multiply"(%arg9, %120) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %122 = "stablehlo.add"(%119, %121) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %123 = "stablehlo.multiply"(%arg14, %96) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %124 = "stablehlo.subtract"(%123, %arg8) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %125 = "stablehlo.multiply"(%arg12, %98) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %126 = "stablehlo.subtract"(%124, %125) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %127 = "stablehlo.broadcast_in_dim"(%90) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %128 = "stablehlo.multiply"(%arg10, %127) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %129 = "stablehlo.add"(%126, %128) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %130 = "stablehlo.multiply"(%115, %129) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %131 = "stablehlo.subtract"(%122, %130) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %132 = "stablehlo.divide"(%131, %104) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %133 = "stablehlo.select"(%107, %132, %arg15) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %134 = "stablehlo.subtract"(%133, %arg15) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %135 = "stablehlo.abs"(%134) : (tensor<20x20xf32>) -> tensor<20x20xf32>
      %136 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %137 = "stablehlo.broadcast_in_dim"(%136) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %138 = "stablehlo.select"(%107, %135, %137) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %139 = "stablehlo.abs"(%101) : (tensor<20x20xf32>) -> tensor<20x20xf32>
      %140 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %141 = "func.call"(%140) <{callee = @integer_pow}> : (tensor<f32>) -> tensor<f32>
      %142 = "stablehlo.broadcast_in_dim"(%141) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %143 = "stablehlo.compare"(%139, %142) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %144 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %145 = "stablehlo.broadcast_in_dim"(%144) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %146 = "stablehlo.multiply"(%arg7, %145) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %147 = "stablehlo.select"(%143, %146, %arg7) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %148 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %149 = "stablehlo.broadcast_in_dim"(%148) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %150 = "stablehlo.multiply"(%101, %149) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %151 = "stablehlo.select"(%143, %150, %101) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %152 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %153 = "stablehlo.broadcast_in_dim"(%152) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %154 = "stablehlo.multiply"(%arg8, %153) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %155 = "stablehlo.select"(%143, %154, %arg8) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %156 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %157 = "stablehlo.broadcast_in_dim"(%156) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %158 = "stablehlo.multiply"(%104, %157) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %159 = "stablehlo.select"(%143, %158, %104) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %160 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %161 = "stablehlo.broadcast_in_dim"(%160) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %162 = "stablehlo.multiply"(%arg13, %161) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %163 = "stablehlo.select"(%143, %162, %arg13) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %164 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %165 = "stablehlo.broadcast_in_dim"(%164) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %166 = "stablehlo.multiply"(%arg14, %165) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %167 = "stablehlo.select"(%143, %166, %arg14) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %168 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %169 = "stablehlo.broadcast_in_dim"(%168) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %170 = "stablehlo.multiply"(%122, %169) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %171 = "stablehlo.select"(%143, %170, %122) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %172 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %173 = "stablehlo.broadcast_in_dim"(%172) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %174 = "stablehlo.multiply"(%129, %173) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %175 = "stablehlo.select"(%143, %174, %129) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %176 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %177 = "stablehlo.broadcast_in_dim"(%176) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %178 = "stablehlo.compare"(%114, %177) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %179 = "stablehlo.and"(%arg1, %178) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
      %180 = "stablehlo.select"(%arg1, %115, %arg2) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %181 = "stablehlo.select"(%arg1, %114, %arg3) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %182 = "stablehlo.select"(%arg1, %93, %arg4) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %183 = "stablehlo.select"(%arg1, %96, %arg5) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %184 = "stablehlo.select"(%arg1, %151, %arg7) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %185 = "stablehlo.select"(%arg1, %159, %arg8) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %186 = "stablehlo.select"(%arg1, %147, %arg9) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %187 = "stablehlo.select"(%arg1, %155, %arg10) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %188 = "stablehlo.select"(%arg1, %163, %arg11) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %189 = "stablehlo.select"(%arg1, %167, %arg12) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %190 = "stablehlo.select"(%arg1, %171, %arg13) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %191 = "stablehlo.select"(%arg1, %175, %arg14) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %192 = "stablehlo.select"(%arg1, %133, %arg15) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      "stablehlo.return"(%179, %180, %181, %182, %183, %90, %184, %185, %186, %187, %188, %189, %190, %191, %192) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    }) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>) -> (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>)
    %75 = "stablehlo.multiply"(%74#1, %32) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %76 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %77 = "stablehlo.broadcast_in_dim"(%76) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %78 = "stablehlo.subtract"(%77, %44) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %79 = "stablehlo.select"(%19, %78, %75) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %80 = "stablehlo.constant"() <{value = dense<0x7F800000> : tensor<f32>}> : () -> tensor<f32>
    %81 = "stablehlo.broadcast_in_dim"(%80) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %82 = "stablehlo.compare"(%7, %81) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %83 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %84 = "stablehlo.broadcast_in_dim"(%83) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %85 = "stablehlo.select"(%82, %84, %79) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %86 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %87 = "stablehlo.broadcast_in_dim"(%86) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %88 = "stablehlo.select"(%14, %87, %85) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    "stablehlo.custom_call"(%88, %6) <{call_target_name = "check.expect_almost_eq", has_side_effect = true}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    "func.return"(%88) : (tensor<20x20xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<20x20xf32>, tensor<1x20xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %3 = "stablehlo.constant"() <{value = dense<"0xBA1452BF375D99C0F4B5E0BF6759743FBA894A403359C83EEC8A24402D46E23FB9B10A402E67493F74CE2EBF17316240AA7E5240173733BF46459F3F7434B1BE15F9B8C09A52DB3F96E05640375A29C09B2C1AC045286F401A638CBFC5B4D1BE0A0B94C01F345E3F2CB0F63D7F889F405FF3B9BE4D7A1540C94BD9BF29DF083FA4FE003F1322E0BE9560FD3F409482C01105FCBF03C781BF47F15D407F27C640E2C7D4BF4953993D57A18EBF6FD3F93F4624084025C352C025113A3DDB3DC2BFCA5FA740C9B613C049A017C03557FB3F34E68840233A093FF2D414C0A46A534045F8DFBFB83461BF7B3FB9C0065898C0F95AACBFD3EDAF3F8D99E43FE884FBBDA538C73E5C3726BF0802B43F7343B1BFE9FB1140116099C056419FC0FC05A43FAFDFFABF0340FFBFE9E6A73F6E4718C0B6C978BF4A675F40A3628E3E90F3E23FD0A842BECB5754C0792446C06F175440E02C51402A8624BFB4B4F0BE963732C08A008F40F6F86EBF88856DC0634F98C04B2F99C06F33273F9ECFA13F6323403F2E0DEC3F8437CEBFAECCC43D5034B8BF169C4E3FDAD741BF82CB45C077792E40AD4CC7C08C0A8DBFFB8509409C04B73F7B31623E13B523C02A48A1C06E49C4C0805E93406FC91940BF7418C06C1300C0700009C0BBFAE63F66B0B4BF31E032BF261BD83F16490740D300D0408FFFC63F3A30A9409382A1BFDC2786401D7D913F7AF4B3C026591BC017618FBEC02A1B40BDD7274062CD2C408ADD4440DAC60F40A2320E40881189C07DD656BFA0E135BF8ABC3FC01E04563FB45B7FC0842B133F67E8933F111A1840E2AADB40D1F88DBED7B601C0C37AB1403F1EB2C096DAD8BF756C064086D2E3BF75E05E40FE930F403EBB59BFD7F9D9BE23C88E3F2BA0A5C05E4A8EC00A2D3F40FD7CFD3FDB1AC7BF1D30AF400B361EC0C51DAA3FBD53444031C44740B2DBC8BFD2E57F40858A24BF99EF54BF1385CEBECA22A43F5986404010DF8EC01A200D403DA539C0A59861C0480B41C06D6C154066BCB23F4F7F27BF308C4ABF37A11740985C24BEFCA74A3FE94E6B3FCE5FCDC03C6481405AC85740ADE838C0C192B9BF5CB2844012621BBF18DA6A40E73CAE3FA4A997BF35DC553F977D9FC05DDF693F66CAFEBF3FCCEEBF8E8ED4BF42D0A04079A0E73F37D2713F0B31483FCD39B73F8FD51B4009B240C0C34093C09DDDAD3FABDD79C03A5B9DC0ED443140B30B83BF92740340EEEA73BF445576BF7FA099C0DFF3644049BF63BFE15CBABC11DB0B4054417940C36050C0411953C0CBB015BE36B9B7BF3DEFAC40BC83B8BFE12DC1BEB14D863F0E51E24060A90D409AB8143F68CB864029195740AB4628BFF930BA3F74C8E0BFC8FD8E3FFD7AB340DD13ADBE3097603F708EA33FA149BC3FCA5AC4BC3202C4C0552C574088B32140FB8DBB3F51CEC1C09A3B1140E3D0FBBF96AA88BEFB7E614053221940D6F2C0C0E45347403752443F25458A407FB6D03F2110EB40DD93843F6F3F234011E3AD4054A2A53E938EBEBF3618D53F42328840E8AF83C002C11FC0EDD27BBE26E112C0012C3F3FB501853F9B4D17BFEB4FD140594C9ABF614F3340C2E4F6BF4BF323405CD728C025167440309B4740674ECC3FAEE725C06750E2BFB1FFF53F05814D3FE0E526C0ADAE92405FD2ADBF52DAA83E4A3B7CC0870519C01222D53FFA7C91BF854F14404E21DD3F23968A3FD17928BF616286BE63EF303EE621B0C08D5F9EBF329E00C068F23BBF3256333F4DFA3AC08F3F51C099F25E407687773FBEE4BC3F1F72554035B60C3F0BFA98C0CB2492BF79918CC0F5B9B43FAB009F407B40DABD3832F1BFE16A95C08A09B03FC48596C027AC87C051F0ACC06A66C7BF09EDE8BF1C41C140147C89405932E6BE5F9C763F90B07B3E6070E03FC33B9DBFBE5C8D40F44880C0DED114405761C0BEF022C33F788D3C4055E6013F658A15C0D07FDDBF3E8004C0329682406D1C693FF8BC1840FB417CC061F7C4BF73D41940E2313440E77653BF7D2A71C08C63434013398BC0DE77E9C062BB39C012CA383F3F93284052EB5B3F6DEE8AC04D7BFEBF18398A40A7D797401F3FF9BFDFD34040D88FB43FF2FDC6BF37F09240BA9B0E406E5B423F48D2723E543FAEC03BAA00C0EAC1FB3F97319DBFB121593FF0B0FA3FC25DE73E87FD22BFEAD804BFDA681F40FBB91BC0C6BCB7BF797AD53F6DF060404E58143F1826A93F74CE893F523B893F62F09C408B44A74043BE48C09FECDA3D"> : tensor<20x20xf32>}> : () -> tensor<20x20xf32>
    %4 = "stablehlo.constant"() <{value = dense<[[3.33433104, -3.0612781, 1.59221566, 1.76121521, -3.06256843, 1.50310075, -3.63711357, -1.06411946, 2.40649104, 1.22511339, -0.406261683, -5.50833511, 0.86351943, 2.61897612, -0.425570458, -1.28150737, 0.478907317, -1.39718246, 3.93083906, -1.35018599]]> : tensor<1x20xf32>}> : () -> tensor<1x20xf32>
    "func.return"(%3, %4) : (tensor<20x20xf32>, tensor<1x20xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %2 = "stablehlo.constant"() <{value = dense<"0x0000803F0000803F0000803F70C5233E0000803F3E64733D0000803F0000803F888DB33E7E565B3E0000803F0000803FB089763F0000803F0000803F0000803F0000803F0000803FAA13A23E0000803F0000803F0000803F0000803F0000803F0000803FF0D7393E0000803F0000803F0000803F9CE73E3F0000803F0000803F381C433E0000803F0000803F0000803F0000803F0000803FCB4FAD3E0000803F0000803F0000803F0000803F5A4BEB3E0000803F0000803F0000803F0000803F8D066C3F0000803F0000803F0000803F44237E3FBD7ECA3C0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F76E8EA3E0000803F0000803F0000803F0000803F0000803F2601C33E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F880D003B0000803F0000803F0000803F0000803FF8524D3F0000803F0000803F0000803F0000803F11AB583F0000803F0000803F0000803F0000803FAC860C3D0000803F0000803FA912643F0000803FC338FC390000803F8933BF3C0000803F0000803F00A02D3F0000803F0000803F0000803F0000803F8CD30F3C0000803F0000803F0000803F87F57E3FA5D0B93E0000803F0000803F0000803F0000803F0000803F0000803FC8DADB3D0000803F71477F3F287AAB3E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FAD22673FF677E03E0000803F0000803FB272713F0000803F0000803F0000803F0000803F0000803F0000803F7BF6973D0000803F7EB62A3F0000803F0000803F0000803F09187F3F0000803F0000803FF2564F3F0000803F0000803F0000803F0000803F0000803F3C24CC3C0000803F0000803F0000803F3F73053F0000803F0000803F0000803F0000803F0000803F5AF4183F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F33C6A83E0000803F0000803F38342A3F0000803F0000803F1E259F3D0000803F0000803F0000803F0000803F0000803F0000803F0000803FFF327F3F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F645E7B3F0000803F0000803FB0A3703D4AF0EA3E0000803F0000803F0000803F47EB033E0000803F0000803F02CD7A3F0000803F822FD33D0000803F0000803F0000803F89A6603F0000803F0000803F80EF1C3F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F683A713F0000803FEC23FB3E0000803F0000803F0000803F0000803F08AA503E0000803F0000803F0000803F0000803F62F5373E0000803F0000803F0000803F74B3633FE813183E0000803F0000803F0000803F0000803FB305B33E0000803F0000803F0000803FC3EF0C3E6E486D3F0000803F5DDD7F3F0000803F0000803F4C946F3F89B7893D0000803F0000803FA5107E3F0000803F0000803F0000803F0000803F0000803FD7B7AE3C0000803FF1CE6B3F0000803FC57F3E3F0000803F0000803F0000803F0000803F0000803FF8B5543E0000803F0000803F0000803F88C3A83E0000803F0000803F0000803FAC975A3E0000803F0000803F0000803F0000803F0000803FDC18E13EE1DD463E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F42ED4D3F0000803F6FFFC33B0000803F0000803F0000803F1669AB3EAE0D773F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FC792733F0000803F0000803F1AEE193F0000803F2DE6903D0000803FCEFA273F0000803F9DD41F3F0000803F0000803F42474C3F0000803F0000803F0000803F0000803F0000803F0000803F0FFB5D3F0000803F0000803F0000803F743F7B3F0000803F0000803F0000803F0000803F0000803F0000803F588ED53D0000803F5B19373E0000803F0000803F378F533F32D37C3F0000803F0000803F6CDD183F0000803F0000803F0000803F62EBFA3E0000803F0000803F0000803F57A0183E0000803F5D74233E1A5EEC3E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F7061663E9E91FA3D0000803F0000803F4FF47F3F0000803F0000803F0000803F"> : tensor<20x20xf32>}> : () -> tensor<20x20xf32>
    "func.return"(%2) : (tensor<20x20xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<f32>) -> tensor<f32>, sym_name = "integer_pow", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<f32>):
    %0 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %1 = "stablehlo.divide"(%0, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "func.return"(%1) : (tensor<f32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

