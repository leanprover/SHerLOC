"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %5:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<20x20xf16>, tensor<20x20xf16>)
    %6 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xf16>
    %7 = "stablehlo.convert"(%5#0) : (tensor<20x20xf16>) -> tensor<20x20xf32>
    %8 = "stablehlo.convert"(%5#1) : (tensor<20x20xf16>) -> tensor<20x20xf32>
    %9 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %10 = "stablehlo.broadcast_in_dim"(%9) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %11 = "stablehlo.compare"(%8, %10) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %12 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %13 = "stablehlo.broadcast_in_dim"(%12) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %14 = "stablehlo.compare"(%7, %13) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %15 = "stablehlo.or"(%11, %14) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %16 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %17 = "stablehlo.broadcast_in_dim"(%16) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %18 = "stablehlo.compare"(%8, %17) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %19 = "stablehlo.compare"(%8, %7) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %20 = "stablehlo.or"(%18, %19) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %21 = "stablehlo.log"(%8) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %22 = "stablehlo.multiply"(%7, %21) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %23 = "stablehlo.subtract"(%22, %8) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %24 = "chlo.lgamma"(%7) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %25 = "stablehlo.subtract"(%23, %24) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %26 = "stablehlo.constant"() <{value = dense<3.40282347E+38> : tensor<f32>}> : () -> tensor<f32>
    %27 = "stablehlo.log"(%26) : (tensor<f32>) -> tensor<f32>
    %28 = "stablehlo.negate"(%27) : (tensor<f32>) -> tensor<f32>
    %29 = "stablehlo.broadcast_in_dim"(%28) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %30 = "stablehlo.compare"(%25, %29) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %31 = "stablehlo.or"(%15, %30) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %32 = "stablehlo.not"(%31) : (tensor<20x20xi1>) -> tensor<20x20xi1>
    %33 = "stablehlo.exponential"(%25) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %34 = "stablehlo.and"(%32, %20) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %35 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %36 = "stablehlo.broadcast_in_dim"(%35) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %37 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %38 = "stablehlo.broadcast_in_dim"(%37) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %39 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %40 = "stablehlo.broadcast_in_dim"(%39) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %41 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %42 = "stablehlo.broadcast_in_dim"(%41) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %43:7 = "stablehlo.while"(%34, %7, %36, %38, %8, %40, %42) ({
    ^bb0(%arg40: tensor<20x20xi1>, %arg41: tensor<20x20xf32>, %arg42: tensor<20x20xf32>, %arg43: tensor<20x20xf32>, %arg44: tensor<20x20xf32>, %arg45: tensor<20x20xf32>, %arg46: tensor<20x20xf32>):
      %225 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
      %226 = "stablehlo.reduce"(%arg40, %225) <{dimensions = array<i64: 0, 1>}> ({
      ^bb0(%arg47: tensor<i1>, %arg48: tensor<i1>):
        %227 = "stablehlo.or"(%arg47, %arg48) : (tensor<i1>, tensor<i1>) -> tensor<i1>
        "stablehlo.return"(%227) : (tensor<i1>) -> ()
      }) : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%226) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg33: tensor<20x20xi1>, %arg34: tensor<20x20xf32>, %arg35: tensor<20x20xf32>, %arg36: tensor<20x20xf32>, %arg37: tensor<20x20xf32>, %arg38: tensor<20x20xf32>, %arg39: tensor<20x20xf32>):
      %201 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %202 = "stablehlo.broadcast_in_dim"(%201) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %203 = "stablehlo.add"(%arg34, %202) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %204 = "stablehlo.divide"(%arg37, %203) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %205 = "stablehlo.multiply"(%arg38, %204) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %206 = "stablehlo.multiply"(%arg35, %arg37) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %207 = "stablehlo.multiply"(%203, %203) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %208 = "stablehlo.divide"(%206, %207) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %209 = "stablehlo.subtract"(%205, %208) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %210 = "stablehlo.add"(%arg39, %209) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %211 = "stablehlo.divide"(%arg37, %203) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %212 = "stablehlo.multiply"(%arg35, %211) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %213 = "stablehlo.add"(%arg36, %212) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %214 = "stablehlo.divide"(%212, %213) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %215 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %216 = "stablehlo.broadcast_in_dim"(%215) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %217 = "stablehlo.compare"(%214, %216) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %218 = "stablehlo.and"(%arg33, %217) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
      %219 = "stablehlo.select"(%arg33, %203, %arg34) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %220 = "stablehlo.select"(%arg33, %212, %arg35) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %221 = "stablehlo.select"(%arg33, %213, %arg36) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %222 = "stablehlo.select"(%arg33, %arg37, %arg37) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %223 = "stablehlo.select"(%arg33, %209, %arg38) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %224 = "stablehlo.select"(%arg33, %210, %arg39) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      "stablehlo.return"(%218, %219, %220, %221, %222, %223, %224) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    }) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>) -> (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>)
    %44 = "stablehlo.multiply"(%43#3, %33) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %45 = "stablehlo.divide"(%44, %7) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %46 = "stablehlo.not"(%20) : (tensor<20x20xi1>) -> tensor<20x20xi1>
    %47 = "stablehlo.and"(%32, %46) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %48 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %49 = "stablehlo.broadcast_in_dim"(%48) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %50 = "stablehlo.subtract"(%49, %7) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %51 = "stablehlo.add"(%8, %50) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %52 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %53 = "stablehlo.broadcast_in_dim"(%52) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %54 = "stablehlo.add"(%51, %53) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %55 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %56 = "stablehlo.broadcast_in_dim"(%55) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %57 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %58 = "stablehlo.broadcast_in_dim"(%57) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %59 = "stablehlo.add"(%8, %58) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %60 = "stablehlo.multiply"(%54, %8) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %61 = "stablehlo.divide"(%59, %60) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %62 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %63 = "stablehlo.broadcast_in_dim"(%62) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %64 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %65 = "stablehlo.broadcast_in_dim"(%64) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %66 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %67 = "stablehlo.broadcast_in_dim"(%66) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %68 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %69 = "stablehlo.broadcast_in_dim"(%68) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %70 = "stablehlo.negate"(%8) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %71 = "stablehlo.multiply"(%61, %70) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %72 = "stablehlo.subtract"(%69, %71) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %73 = "stablehlo.divide"(%72, %60) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %74 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %75:15 = "stablehlo.while"(%47, %61, %63, %50, %54, %74, %59, %60, %56, %8, %65, %67, %69, %70, %73) ({
    ^bb0(%arg16: tensor<20x20xi1>, %arg17: tensor<20x20xf32>, %arg18: tensor<20x20xf32>, %arg19: tensor<20x20xf32>, %arg20: tensor<20x20xf32>, %arg21: tensor<f32>, %arg22: tensor<20x20xf32>, %arg23: tensor<20x20xf32>, %arg24: tensor<20x20xf32>, %arg25: tensor<20x20xf32>, %arg26: tensor<20x20xf32>, %arg27: tensor<20x20xf32>, %arg28: tensor<20x20xf32>, %arg29: tensor<20x20xf32>, %arg30: tensor<20x20xf32>):
      %195 = "stablehlo.constant"() <{value = dense<2.000000e+03> : tensor<f32>}> : () -> tensor<f32>
      %196 = "stablehlo.compare"(%arg21, %195) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %197 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
      %198 = "stablehlo.reduce"(%arg16, %197) <{dimensions = array<i64: 0, 1>}> ({
      ^bb0(%arg31: tensor<i1>, %arg32: tensor<i1>):
        %200 = "stablehlo.or"(%arg31, %arg32) : (tensor<i1>, tensor<i1>) -> tensor<i1>
        "stablehlo.return"(%200) : (tensor<i1>) -> ()
      }) : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      %199 = "stablehlo.and"(%196, %198) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%199) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg1: tensor<20x20xi1>, %arg2: tensor<20x20xf32>, %arg3: tensor<20x20xf32>, %arg4: tensor<20x20xf32>, %arg5: tensor<20x20xf32>, %arg6: tensor<f32>, %arg7: tensor<20x20xf32>, %arg8: tensor<20x20xf32>, %arg9: tensor<20x20xf32>, %arg10: tensor<20x20xf32>, %arg11: tensor<20x20xf32>, %arg12: tensor<20x20xf32>, %arg13: tensor<20x20xf32>, %arg14: tensor<20x20xf32>, %arg15: tensor<20x20xf32>):
      %91 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %92 = "stablehlo.add"(%arg6, %91) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      %93 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %94 = "stablehlo.broadcast_in_dim"(%93) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %95 = "stablehlo.add"(%arg4, %94) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %96 = "stablehlo.constant"() <{value = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %97 = "stablehlo.broadcast_in_dim"(%96) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %98 = "stablehlo.add"(%arg5, %97) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %99 = "stablehlo.broadcast_in_dim"(%92) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %100 = "stablehlo.multiply"(%95, %99) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %101 = "stablehlo.multiply"(%arg7, %98) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %102 = "stablehlo.multiply"(%arg9, %100) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %103 = "stablehlo.subtract"(%101, %102) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %104 = "stablehlo.multiply"(%arg8, %98) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %105 = "stablehlo.multiply"(%arg10, %100) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %106 = "stablehlo.subtract"(%104, %105) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %107 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %108 = "stablehlo.broadcast_in_dim"(%107) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %109 = "stablehlo.compare"(%106, %108) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %110 = "stablehlo.divide"(%103, %106) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %111 = "stablehlo.subtract"(%arg2, %110) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %112 = "stablehlo.divide"(%111, %110) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %113 = "stablehlo.abs"(%112) : (tensor<20x20xf32>) -> tensor<20x20xf32>
      %114 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %115 = "stablehlo.broadcast_in_dim"(%114) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %116 = "stablehlo.select"(%109, %113, %115) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %117 = "stablehlo.select"(%109, %110, %arg2) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %118 = "stablehlo.multiply"(%arg13, %98) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %119 = "stablehlo.subtract"(%118, %arg7) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %120 = "stablehlo.multiply"(%arg11, %100) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %121 = "stablehlo.subtract"(%119, %120) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %122 = "stablehlo.broadcast_in_dim"(%92) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %123 = "stablehlo.multiply"(%arg9, %122) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %124 = "stablehlo.add"(%121, %123) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %125 = "stablehlo.multiply"(%arg14, %98) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %126 = "stablehlo.subtract"(%125, %arg8) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %127 = "stablehlo.multiply"(%arg12, %100) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %128 = "stablehlo.subtract"(%126, %127) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %129 = "stablehlo.broadcast_in_dim"(%92) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %130 = "stablehlo.multiply"(%arg10, %129) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %131 = "stablehlo.add"(%128, %130) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %132 = "stablehlo.multiply"(%117, %131) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %133 = "stablehlo.subtract"(%124, %132) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %134 = "stablehlo.divide"(%133, %106) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %135 = "stablehlo.select"(%109, %134, %arg15) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %136 = "stablehlo.subtract"(%135, %arg15) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %137 = "stablehlo.abs"(%136) : (tensor<20x20xf32>) -> tensor<20x20xf32>
      %138 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %139 = "stablehlo.broadcast_in_dim"(%138) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %140 = "stablehlo.select"(%109, %137, %139) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %141 = "stablehlo.abs"(%103) : (tensor<20x20xf32>) -> tensor<20x20xf32>
      %142 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %143 = "func.call"(%142) <{callee = @integer_pow}> : (tensor<f32>) -> tensor<f32>
      %144 = "stablehlo.broadcast_in_dim"(%143) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %145 = "stablehlo.compare"(%141, %144) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %146 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %147 = "stablehlo.broadcast_in_dim"(%146) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %148 = "stablehlo.multiply"(%arg7, %147) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %149 = "stablehlo.select"(%145, %148, %arg7) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %150 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %151 = "stablehlo.broadcast_in_dim"(%150) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %152 = "stablehlo.multiply"(%103, %151) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %153 = "stablehlo.select"(%145, %152, %103) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %154 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %155 = "stablehlo.broadcast_in_dim"(%154) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %156 = "stablehlo.multiply"(%arg8, %155) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %157 = "stablehlo.select"(%145, %156, %arg8) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %158 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %159 = "stablehlo.broadcast_in_dim"(%158) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %160 = "stablehlo.multiply"(%106, %159) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %161 = "stablehlo.select"(%145, %160, %106) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %162 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %163 = "stablehlo.broadcast_in_dim"(%162) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %164 = "stablehlo.multiply"(%arg13, %163) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %165 = "stablehlo.select"(%145, %164, %arg13) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %166 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %167 = "stablehlo.broadcast_in_dim"(%166) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %168 = "stablehlo.multiply"(%arg14, %167) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %169 = "stablehlo.select"(%145, %168, %arg14) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %170 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %171 = "stablehlo.broadcast_in_dim"(%170) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %172 = "stablehlo.multiply"(%124, %171) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %173 = "stablehlo.select"(%145, %172, %124) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %174 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %175 = "stablehlo.broadcast_in_dim"(%174) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %176 = "stablehlo.multiply"(%131, %175) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %177 = "stablehlo.select"(%145, %176, %131) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %178 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %179 = "stablehlo.broadcast_in_dim"(%178) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %180 = "stablehlo.compare"(%116, %179) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %181 = "stablehlo.and"(%arg1, %180) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
      %182 = "stablehlo.select"(%arg1, %117, %arg2) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %183 = "stablehlo.select"(%arg1, %116, %arg3) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %184 = "stablehlo.select"(%arg1, %95, %arg4) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %185 = "stablehlo.select"(%arg1, %98, %arg5) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %186 = "stablehlo.select"(%arg1, %153, %arg7) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %187 = "stablehlo.select"(%arg1, %161, %arg8) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %188 = "stablehlo.select"(%arg1, %149, %arg9) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %189 = "stablehlo.select"(%arg1, %157, %arg10) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %190 = "stablehlo.select"(%arg1, %165, %arg11) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %191 = "stablehlo.select"(%arg1, %169, %arg12) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %192 = "stablehlo.select"(%arg1, %173, %arg13) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %193 = "stablehlo.select"(%arg1, %177, %arg14) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %194 = "stablehlo.select"(%arg1, %135, %arg15) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      "stablehlo.return"(%181, %182, %183, %184, %185, %92, %186, %187, %188, %189, %190, %191, %192, %193, %194) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    }) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>) -> (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>)
    %76 = "stablehlo.multiply"(%75#1, %33) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %77 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %78 = "stablehlo.broadcast_in_dim"(%77) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %79 = "stablehlo.subtract"(%78, %45) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %80 = "stablehlo.select"(%20, %79, %76) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %81 = "stablehlo.constant"() <{value = dense<0x7F800000> : tensor<f32>}> : () -> tensor<f32>
    %82 = "stablehlo.broadcast_in_dim"(%81) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %83 = "stablehlo.compare"(%8, %82) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %84 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %85 = "stablehlo.broadcast_in_dim"(%84) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %86 = "stablehlo.select"(%83, %85, %80) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %87 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %88 = "stablehlo.broadcast_in_dim"(%87) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %89 = "stablehlo.select"(%15, %88, %86) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %90 = "stablehlo.convert"(%89) : (tensor<20x20xf32>) -> tensor<20x20xf16>
    "stablehlo.custom_call"(%90, %6) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<20x20xf16>, tensor<20x20xf16>) -> ()
    "func.return"(%90) : (tensor<20x20xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<20x20xf16>, tensor<20x20xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %3 = "stablehlo.constant"() <{value = dense<"0xB2C482ACAE4199BF7BB54142193E7E3070C6684260C124C4FE4399447EBC0F37FA380F3E9CBE0DBA2244DB44913973C16D3B503330BDA736F033AAC297BC9343F3BDEFBA4741EDC05140E0BC4CC3AFC1FB46B5C6B12E0243443C59B964BB0DBC2D41CC417E4018BC1AB99039153EC64110C055C21CC430BBE03C43BCC4BDACC0A0BE39443FBCEA4184BE343D4B353BBD5DC11A40503C65C39A458EBD5CC5C540A33EC139F2B9D340BEBE37BF79420F3A763F69C33D42EB3723B58831A5457043653DA6C11E43E1C328C6A4C400BEEE3A48415F430E44F741AB3D34C2AE40EB3ED74456BECF3CA7C021427540CCA6E63E4BC6DAC07DC4D63E944480C0043895C2DF233DC0D1C2C43FBABCD044B4C08B3CE9C79B468CC452C5A445CE4233371745703769B4544062C1B3475C39C7C1B445A33C05402B40C13FE9B6BA3C58C3B8462A433BC27CB970BD9A4422397C43E5C06645DAC18D445AC835B91BC6F33DEA33FB444AC219BE70BCABB46A3BA93D8B31F3BEF0C24C37C7BD74C5C3C067BEBB3D26BC07C0B4ACB2C2F94305C592C1D5386737C6BEDE436A3E914298C0E6B5FC3D94C00DC475C322C067BEC7C512B6C0C800C48946394462C08344C43ED4C0F4C0F33DCEB6F03A1CBD18409131A2456F3B74AC2A3A33B7D24543B89E3DE2C342C1AABD76B99B3563BC6743BF3CF8BFA730C93D864604BE9AC0F23BD7C321B6A4420BBEE3C65045A6C514410D4541328AC00F3C56AEC5C0C642FF4494B92AC0D3C4223993415F3E5CC1D8383845934380C2A640CB3FD1BF9C40AD3DCF4306315DB6C8404FBF654770443EC09DB8023FA93EBE441B385EC4EEC620BD1AC35CB853C3C642EA4353C59C453DBF6DBA3C3630BF15C4F5453F4288C433C3103DA13DF239D63E68C060C104B84941B3ACC940A2BEF641CD43CEB90E349A4169C37CC25D409E4538C408C41635EA3529C78DBD0B430E3D67C4873D43395938A0C17DB80CC1F9C4F4BFDEBE8EC3AC40184529C419BC734220B533B71D3585B23D3BE1C7EAC2FA4304412C388640004288BC5D1FDFC205453D3EC0C380C2AF3F5C4480C6904403390DC7693DBC3867BB8A4047BF1D38FE4336446A3D0DB66046713C3A3CFFC4"> : tensor<20x20xf16>}> : () -> tensor<20x20xf16>
    %4 = "stablehlo.constant"() <{value = dense<"0x44438540C2B869C2CA2FF6BD4F971BBC05C47CC6B02E15445EC44242FDB8E3C4244063C35EBC11C287B8473B2CC38BBBB13C39A93D3940396FBDC4BF32BD1D3D9A39903DD3BEABC0B4BCC1C4D4C482B26CBEF93F37C14E3F943E574634A9BBC43ABCE041A2B065AFAB44BE3F453A0EC3BA4415BBB3C4434124C3DFC0F6406246FDBD1F2F00C631B7944474C670C44C4383BA414133457CBD573E123D3CB94441652FE3338B450644E8BCA8C579C1743FEEB9EFB782396AC44447C442F43A2CBF3EC00246B4BD7EBFAF419EB909448044644014347BBF3645014507443C4092435BC65BC052BE32B275C091B50A3F78BB0B419437F13A5543AFC26446BEB3304207C47AC1CAC551C350C41EBDEB450EBC943778BF08C525C04F44EE41AEBE5043344475BF22C22CC12FC383BD31409A3C093791A5CAC1113D15BF10C0CD3E26C331B0393D7D3C1542B143533EF8320A3B6DC15840E4450BC577B928C521C40F3B90BAD93A4DBF5642602E2CC4B243B83FCA4231B5554843381247AAAD1CC3B7B82FBE673DCFBF1E44CF38DEC42FC58DBCDDC0BF423030C9C0314253410844FDBD343F86C48FC0374444BD513E9EBE8843E4403EB55AB86038FD40743F2244E5C3FE34FFC00E2DEB3E85BDAA45E0C0E1413BC6D7C0A7BD71BECFC12DBB023DBCB7A9C1AEBC613693BEC1C0963C1D2FE6BE19BA233BDCBA893D33B04BB75F40C7C31B41F9436EC585C1B0C684331CBD20434E3844C2EAC211C22BB98FC448C6E84396C31735843B673FAA422641A0B4B73892C2D741D24235BF6AC1B138CEBB88341AA5003C9C3CC34290C1F7C19CC0422E8DB69B3E4EC2414102BEF435FBBC7E3937420C408DB1D13ADB3852BB1643DAC01BC17E3FED3EF54112C1DEC4BBBE12C192BBE0C61CC6D84462473C3C1141EB3F17C40440D6403BC009347CC43F28EDC79E3CC04197BC34C32DC6F437FD43B5BD2545D63F7BBC3E36D63FB8B9643DB1C1A9C269B1A53E373A1C4436C2DABD18B8E9AA40C13ABDBC3DBDC394432AC2423C0B448D4209B689410442DDBE0A43FBC1233CC1BC724092BA07418A445DC2E54108B7583D02BF60BA1B41B73D913DE73EB0C6A1C451470A407042473994448F48"> : tensor<20x20xf16>}> : () -> tensor<20x20xf16>
    "func.return"(%3, %4) : (tensor<20x20xf16>, tensor<20x20xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %2 = "stablehlo.constant"() <{value = dense<"0x003C003C003C003C003C003C003C003C003C003C003C003C003CD839003C003C7E2B003C003C003C003CF93B003C003C8034003C003C9332003C003C003C913B003C003C003C003C003C003C003C003C003C003C003C8E3ACD32003C003C003C003C9C36003C003C003C302D6539003C003C003C003C003C003C003C003C003C003C003C003C003C003C003C003C003C003C6334C81E003CEB3B003C003CAF35DC3B1B39003C8430003C003C003C6B2E003C003CC93B003C003CE317FF3B003C003C003C003C003C003C003C003C22204338FF3B003CCC2E1524003C8C37F62C003C003C003C003C003C003C003C003C003C003C003C562D003C003C003C003C003C003C003C003C003C003C003C003C003C003C003C003C71393538003CAF39BB1A003C003C003C003C003C003CFC3BB939003C003CF538003C003C003C003C003C003C003C003CAF38E12E003C003C003C003C7533003C003C003C003C222D003C003C003C003C003C003CE7291A22003C003CB401003C003C003C003C003C003C003C003C003CF93B003C003C003C003C003C003C003C7D37003C003C003C003C003C003C003C003C003C003C003C003C003C003C003CB23A9A35003C003C1B3B003C1A3B003C003C8508003CE229003C003C003C003C003C003C003C003C003C003C3234003C003C4736003C003C003C003C003C003C003C003C003C003C003C003C003C003C003C003C003C003CBA38003C003C003C003C003C003C003C003CF23B0E39003C0723373B003C003C003C6E32003C003C003CF83B003C003C003C003C003C3439003C003C003CE53B003C982C003C003C003C003C003C003CD337C73A003CFF3B003C003C421D003C003CE53B313A003C003C003C003C003C003C003C003C003C2323003C1536003C003CBD3A003C003CF83B003C003C003CFB3B003C003C003C003C003C003C003C5221003C003CC237ED2A003C003C003C003C003C003C003CD92F003C003C003C003C003C003C632A003CB124003C003CC4362034003CBF34BE36003C4103003CF83B003C003C003C3E347436003C163A003C003C003C003C003C4B39003C362C003C003C7416003C5E3B8E38F921003C"> : tensor<20x20xf16>}> : () -> tensor<20x20xf16>
    "func.return"(%2) : (tensor<20x20xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<f32>) -> tensor<f32>, sym_name = "integer_pow", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<f32>):
    %0 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %1 = "stablehlo.divide"(%0, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "func.return"(%1) : (tensor<f32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

