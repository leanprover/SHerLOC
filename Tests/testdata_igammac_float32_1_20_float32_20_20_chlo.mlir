"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %5:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x20xf32>, tensor<20x20xf32>)
    %6 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xf32>
    %7 = "stablehlo.broadcast_in_dim"(%5#0) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x20xf32>) -> tensor<20x20xf32>
    %8 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %9 = "stablehlo.broadcast_in_dim"(%8) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %10 = "stablehlo.compare"(%5#1, %9) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %11 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %12 = "stablehlo.broadcast_in_dim"(%11) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %13 = "stablehlo.compare"(%7, %12) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %14 = "stablehlo.or"(%10, %13) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %15 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %16 = "stablehlo.broadcast_in_dim"(%15) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %17 = "stablehlo.compare"(%5#1, %16) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %18 = "stablehlo.compare"(%5#1, %7) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %19 = "stablehlo.or"(%17, %18) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %20 = "stablehlo.log"(%5#1) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %21 = "stablehlo.multiply"(%7, %20) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %22 = "stablehlo.subtract"(%21, %5#1) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %23 = "chlo.lgamma"(%7) : (tensor<20x20xf32>) -> tensor<20x20xf32>
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
    %42:7 = "stablehlo.while"(%33, %7, %35, %37, %5#1, %39, %41) ({
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
    %44 = "stablehlo.divide"(%43, %7) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %45 = "stablehlo.not"(%19) : (tensor<20x20xi1>) -> tensor<20x20xi1>
    %46 = "stablehlo.and"(%31, %45) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %47 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %48 = "stablehlo.broadcast_in_dim"(%47) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %49 = "stablehlo.subtract"(%48, %7) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %50 = "stablehlo.add"(%5#1, %49) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %51 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %52 = "stablehlo.broadcast_in_dim"(%51) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %53 = "stablehlo.add"(%50, %52) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %54 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %55 = "stablehlo.broadcast_in_dim"(%54) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %56 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %57 = "stablehlo.broadcast_in_dim"(%56) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %58 = "stablehlo.add"(%5#1, %57) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %59 = "stablehlo.multiply"(%53, %5#1) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %60 = "stablehlo.divide"(%58, %59) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %61 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %62 = "stablehlo.broadcast_in_dim"(%61) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %63 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %64 = "stablehlo.broadcast_in_dim"(%63) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %65 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %66 = "stablehlo.broadcast_in_dim"(%65) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %67 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %68 = "stablehlo.broadcast_in_dim"(%67) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %69 = "stablehlo.negate"(%5#1) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %70 = "stablehlo.multiply"(%60, %69) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %71 = "stablehlo.subtract"(%68, %70) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %72 = "stablehlo.divide"(%71, %59) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %73 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %74:15 = "stablehlo.while"(%46, %60, %62, %49, %53, %73, %58, %59, %55, %5#1, %64, %66, %68, %69, %72) ({
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
    %82 = "stablehlo.compare"(%5#1, %81) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %83 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %84 = "stablehlo.broadcast_in_dim"(%83) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %85 = "stablehlo.select"(%82, %84, %79) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %86 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %87 = "stablehlo.broadcast_in_dim"(%86) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %88 = "stablehlo.select"(%14, %87, %85) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    "stablehlo.custom_call"(%88, %6) <{call_target_name = "check.expect_almost_eq", has_side_effect = true}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    "func.return"(%88) : (tensor<20x20xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x20xf32>, tensor<20x20xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[1.36930537, -1.06475282, -1.10180056, 0.632253647, -1.85677385, 0.1495893, 0.756189227, 3.17537475, -3.76900983, 1.28371584, -5.5223279, -1.53344905, 0.191576734, -1.65628433, 3.71646309, 1.38363183, -1.56151712, -0.758981466, -0.340194583, -2.37838793]]> : tensor<1x20xf32>}> : () -> tensor<1x20xf32>
    %4 = "stablehlo.constant"() <{value = dense<"0xEEB2AF3F0A0DB2BFB690A2C05E059040D8D61EC014FC1440BC2BB84009402F404E3404C024451CC0FB92CA4055A08CC028ADD4400C69494046A25C40EA09BEC06CD43B3EE6C02540552B7CC0B4E41DC041B9B0C085AC2CBFE32AD5C0CDE85B4077EDDA3E04173DBF6C4817BE6C6D78C09428FC3E4C6A8D3F2A200241EB8D354014551B3FA425F0BFC48F86409D49D240F1D0D3C09F80BCBF81D3E53F9E14E9C054EB7E40BB91A3C03D8C2040ED5BF4BF5C4C65407A3BA440663371C03FE882C08F7217C0D8A3ED40DC3B70C0D7D2C840E291E4BEE38E2D40FF2F7C404CDA74C00FB78B40CE54C0C03A0369400492A5C0BF9E4440743C893F5DFB8B3F2D674B40260DE73E0BAD2D4045318040C12677409F113BBF4A761EC0D780C940299E9C3FA829BEBF30F666C01C0692BECDAE61BF6AF0853FA7500F3F51B1A03E834C0DC0DD653440A473FBBFB10C10400D71EFBEF4A5DBBFC89964C01B4D523F0A258A3FD38B363ECB325940064EBBBF9608F63F19F4B2402B19593E72DD5B3FCC0C85C07232483F5E9A1B3F8B8103C0BA7FC03FDEE09CC08B6D2B404CC58DBFE082264059CFFC3FE983F8BFDED4B34011758CBFA232044088861940F37E6FBF071B2C40CBDBACC0C151EB3FF775CB3EFEF6E3BE017736C0F278B9BF1AD101C0FDD52940FBF75A3E8EECB9BFBBA027C0F9C5CA3F1295ED3EAFA872C04E076C408B64A83F5313CABF761F80BF4F3E674012E8B23F5F1E92C0AF34F0BF6AF21BC077373640485F3FC0C4658EBF79F20AC167B007404DCB093F43D5CB3B58DCAEC051411B40EC8AC2BFF24755C0D0A70EBF65A5953FF2060540E30197BE6B808DBFB3E31340E535C93F2EE1E73E23870CBF3E1E863F6295733FCB03A34006692BC0CB2C7940FE7182C083C8023FD5A52740215B273F2B2011C1B50DC1BE5A67A2BF011D77BFA05CE8BF0ADDAF3FE31234C08FA08340DB6DDABE55B830404BE73FBEEB8481C0286D19C0AA838F3FE0BB4DBFB9358ABE923586400FBA7CC0212FA4BFE1DB1CC0C7F076C01F889D3B39E6B44078963D3FF0F96ABF8F7BAB4098B459C047AC04BE2161B03F7F9A0CC0C6B5073F4F83CEC0C5D48D40D098A6C00203944058AFCBBF14619C40FB7E1D40DFDFBB401442D0BF902986BF5B04A13ED82DB5BFD4089440CC4D7640E95B8640DC38773F0CB45B3F2D48F73F314C45BE691C15C032D35440021D6BC0A92744407D51BC3E258E493F7D9C594056A8D03EE2E08D401AFAB6C0BEFCB84082CD0540236CB1C0B4FF56405D67D93FA40E7140DB81853FCCFB51406F5801C0CD885640C8CC17403F7B16C0F21689C04F5CEAC0812E24C08873B83ECDBAB03F518A57C024DF41C09C5440C07C2ACB3F20C72740E2A85A3D380B42C067B6F5BFCC4AA2BF5766AB40824492C0692AA3C0DD11A4BF7AA92040698BD340741C90BEE316923F79A168C0F3148340DAFDC440C91BA03FC29D184038B4FB3FFB0C0D3F6466E63F0C688CBF8F2C19BF99AE27BF37309540980D82C079AC3CC0A73E58BE95E9CA3F4F9A8D3FA9AD4CC0F9C61040387F3A40E120A63F0818C1BFC77F31C0FB414FC069E116BE66A4C13E8086C43F5030C03F0D6744C06CFD2EBFA7C10340C1C3C9BF22B5863F70A3ACBF620D4140233D003FBC7EFDBD1435A240EC5179C058033C4033DB0F4025BE94C0740A7F40BF9C453E639DB7C0EEA0E1BF96B4244076FCFC3FB3030FC03D2EB5405C256D4018088C4001647ABFD667DF40B970A93FF3EB83BFE68316C064749840DDD2C03DB3A0224094708F3F69784ABEC8E4223EF9F79BBF47E109C01C3EAF403E811F40B1F7EA3ECBD7B240901F2C3C3EF3E4BF60C154401E2B32BFC9D185BFB07963BF36EE7EC0BBA780BFF1FC71BEF6B9E83F43729ABFD655CF3F216B39C063A31AC0510F2F40AB5288C05E0A09C09D2BB9BF388A76BEB3D1D93E986563BE4A8CA9C0E52021C06C1ACB3F28A3A4C00EE8D23EBA53F23F783F50407113E83FEB7C63C0E6ECF73FDBC19140DB7BB1BE6B5504C0AC31B23F438C93BF0F8C84C000A14E3F735C364065B0F23FF320C93E47B576401EEF2440A7E3B5C062779B40B910F13F1E33C93F938D86BEE4086CC0097304C0241FEABF23E409402F7FD14054FFB1BCD3A5F3BF016C44C05B2D61407BA88A40BF7F8ABFFBF12DC08791873F6A11B93E8C952D40A2FA4ABF4294C53F6C951940BB5747C07372873F5E0892BF21079DBF63F6B7BD7F6BB23F4E8850BE"> : tensor<20x20xf32>}> : () -> tensor<20x20xf32>
    "func.return"(%3, %4) : (tensor<1x20xf32>, tensor<20x20xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %2 = "stablehlo.constant"() <{value = dense<"0xDD6AC53E0000803F0000803FD9D6893B0000803FFC3CC23BC891D63AF4D9063F0000803F0000803F0000803F0000803FCF4C5D380000803F4080F93E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F3407593C0000803F0000803F0000803F0000803F0000803F8689E43E0000803F0000803FD0D6C23D0000803F998FAD3E841A603B0000803F0000803F0000803F0000803FB49E1A3D0000803F0000803F0000803F0000803F6CE057390000803F0000803F0000803F1B879E3A0000803F0000803F0000803F0000803F662DC63E0000803F0000803F0000803F0000803F0000803F8417B23D0000803F0000803FBBBE8F3C0000803F54426E3B6F4A263C60D5953E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F2DEADF3D0000803F0000803F0000803F0000803F0000803F1075A53E1A786C3F0000803F2A8D673D0000803F0000803FE88533390000803FE0447B3F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F8796073D0000803F0000803FEBF9F63A0000803F0000803FE925113E0000803F0000803F0000803F0000803F889D7F3F0000803F0000803F0000803F0000803F0000803F507D693F0000803F0000803F63BFD63D0000803F0000803FCD66683C0210613F0000803F0000803F0000803F0000803F0000803F0000803F0000803FF0C2DE3D0000803F0000803F0000803F0000803F62593D3F0000803F0000803F60E9243D0000803F0000803F0000803FE764683F0000803F0000803F0000803F0000803F05C0B23C0000803F0000803F5317023F0000803F0000803F0000803F0000803F0000803F0000803F0000803F1825A73E0000803F0000803F0000803F0000803F0000803FDC5DB53E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FF05FFE3C0000803F0000803F0000803F0000803FBC9E043F5C92EE3A4472783F0000803F107A103C0000803F0000803F4A43EA3C0000803FADFC7E3F0000803F0000803F0000803F0000803F0000803F7A5B853C0000803F0000803F0000803F0000803F04D9073E0000803F75263D3E0000803F9B6FD93C0000803F0000803FA7715C3C0000803F0000803FEE9F903D0000803F0000803F0000803F0000803F7C1F843D0000803F0000803F0000803F0000803F6725043C0000803FD3D5C53E0000803FA053233D0000803F0000803F0000803F0000803F64FD3B3F0000803F0000803F0000803F0000803F0000803FB819C43E0000803F0000803F0000803F0000803F4BBA853B745A623F0000803F0000803F0000803F0000803F0000803F0000803F0000803F52CB333F32E7573B0000803F0000803F0000803F0000803F524AA13B0000803F0000803F385C8A3D0000803FC09C423C0000803F0000803F0000803FDC378C3C0000803F0000803F0000803F0000803FC3F2753F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FB4DCF33E0000803F58E8923C0000803F0000803F0000803F0000803F0000803F0000803F6BC5603B0000803F0000803F1104673C0000803F0000803F0000803F0000803F40591A3D0000803F0000803F0000803F0000803F82A3193C0000803FF7B7BF3D0000803F5AE4B73C0000803F0000803F4B0BFD3C0000803F0000803FD98D993C0000803F0000803F0000803F0000803F1D84703F0000803F0000803F035CC33A0000803FD06AC63DC605FF3AFFFF7F3F0000803F6D0B773D0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FE6B1053F0000803F0000803F0000803F0000803F0000803F90260D3E0000803FAEA7073FBE238A3E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F8235493B1DB3C53D09B27E3F0000803FC59CF63D0000803F0000803F16086B3C0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FD676463C0000803F0000803F0000803F85556D3F0000803FBDCFD93D0000803F0000803FCD4DF13B0000803F132A773F0000803F0000803F0000803F0000803F0000803F"> : tensor<20x20xf32>}> : () -> tensor<20x20xf32>
    "func.return"(%2) : (tensor<20x20xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<f32>) -> tensor<f32>, sym_name = "integer_pow", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<f32>):
    %0 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %1 = "stablehlo.divide"(%0, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "func.return"(%1) : (tensor<f32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

