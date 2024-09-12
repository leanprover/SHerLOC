"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %5:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<20x20xf16>, tensor<20x20xf16>)
    %6 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xf16>
    %7 = "stablehlo.convert"(%5#0) : (tensor<20x20xf16>) -> tensor<20x20xf32>
    %8 = "stablehlo.convert"(%5#1) : (tensor<20x20xf16>) -> tensor<20x20xf32>
    %9 = "stablehlo.compare"(%7, %7) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %10 = "stablehlo.compare"(%8, %8) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %11 = "stablehlo.or"(%9, %10) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %12 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %13 = "stablehlo.broadcast_in_dim"(%12) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %14 = "stablehlo.compare"(%8, %13) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %15 = "stablehlo.constant"() <{value = dense<0x7F800000> : tensor<f32>}> : () -> tensor<f32>
    %16 = "stablehlo.broadcast_in_dim"(%15) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %17 = "stablehlo.compare"(%8, %16) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %18 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %19 = "stablehlo.broadcast_in_dim"(%18) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %20 = "stablehlo.compare"(%8, %19) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %21 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %22 = "stablehlo.broadcast_in_dim"(%21) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %23 = "stablehlo.compare"(%7, %22) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %24 = "stablehlo.or"(%20, %23) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %25 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %26 = "stablehlo.broadcast_in_dim"(%25) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %27 = "stablehlo.compare"(%8, %26) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %28 = "stablehlo.compare"(%8, %7) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %29 = "stablehlo.and"(%27, %28) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %30 = "stablehlo.log"(%8) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %31 = "stablehlo.multiply"(%7, %30) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %32 = "stablehlo.subtract"(%31, %8) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %33 = "chlo.lgamma"(%7) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %34 = "stablehlo.subtract"(%32, %33) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %35 = "stablehlo.constant"() <{value = dense<3.40282347E+38> : tensor<f32>}> : () -> tensor<f32>
    %36 = "stablehlo.log"(%35) : (tensor<f32>) -> tensor<f32>
    %37 = "stablehlo.negate"(%36) : (tensor<f32>) -> tensor<f32>
    %38 = "stablehlo.broadcast_in_dim"(%37) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %39 = "stablehlo.compare"(%34, %38) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %40 = "stablehlo.exponential"(%34) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %41 = "stablehlo.or"(%14, %24) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %42 = "stablehlo.or"(%41, %39) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %43 = "stablehlo.or"(%42, %11) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %44 = "stablehlo.not"(%43) : (tensor<20x20xi1>) -> tensor<20x20xi1>
    %45 = "stablehlo.and"(%44, %29) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %46 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %47 = "stablehlo.broadcast_in_dim"(%46) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %48 = "stablehlo.subtract"(%47, %7) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %49 = "stablehlo.add"(%8, %48) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %50 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %51 = "stablehlo.broadcast_in_dim"(%50) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %52 = "stablehlo.add"(%49, %51) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %53 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %54 = "stablehlo.broadcast_in_dim"(%53) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %55 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %56 = "stablehlo.broadcast_in_dim"(%55) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %57 = "stablehlo.add"(%8, %56) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %58 = "stablehlo.multiply"(%52, %8) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %59 = "stablehlo.divide"(%57, %58) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %60 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %61 = "stablehlo.broadcast_in_dim"(%60) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %62 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %63 = "stablehlo.broadcast_in_dim"(%62) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %64 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %65 = "stablehlo.broadcast_in_dim"(%64) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %66 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %67 = "stablehlo.broadcast_in_dim"(%66) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %68 = "stablehlo.negate"(%8) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %69 = "stablehlo.multiply"(%59, %68) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %70 = "stablehlo.subtract"(%67, %69) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %71 = "stablehlo.divide"(%70, %58) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %72 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %73:15 = "stablehlo.while"(%45, %59, %61, %48, %52, %72, %57, %58, %54, %8, %63, %65, %67, %68, %71) ({
    ^bb0(%arg32: tensor<20x20xi1>, %arg33: tensor<20x20xf32>, %arg34: tensor<20x20xf32>, %arg35: tensor<20x20xf32>, %arg36: tensor<20x20xf32>, %arg37: tensor<f32>, %arg38: tensor<20x20xf32>, %arg39: tensor<20x20xf32>, %arg40: tensor<20x20xf32>, %arg41: tensor<20x20xf32>, %arg42: tensor<20x20xf32>, %arg43: tensor<20x20xf32>, %arg44: tensor<20x20xf32>, %arg45: tensor<20x20xf32>, %arg46: tensor<20x20xf32>):
      %234 = "stablehlo.constant"() <{value = dense<2.000000e+03> : tensor<f32>}> : () -> tensor<f32>
      %235 = "stablehlo.compare"(%arg37, %234) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %236 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
      %237 = "stablehlo.reduce"(%arg32, %236) <{dimensions = array<i64: 0, 1>}> ({
      ^bb0(%arg47: tensor<i1>, %arg48: tensor<i1>):
        %239 = "stablehlo.or"(%arg47, %arg48) : (tensor<i1>, tensor<i1>) -> tensor<i1>
        "stablehlo.return"(%239) : (tensor<i1>) -> ()
      }) : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      %238 = "stablehlo.and"(%235, %237) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%238) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg17: tensor<20x20xi1>, %arg18: tensor<20x20xf32>, %arg19: tensor<20x20xf32>, %arg20: tensor<20x20xf32>, %arg21: tensor<20x20xf32>, %arg22: tensor<f32>, %arg23: tensor<20x20xf32>, %arg24: tensor<20x20xf32>, %arg25: tensor<20x20xf32>, %arg26: tensor<20x20xf32>, %arg27: tensor<20x20xf32>, %arg28: tensor<20x20xf32>, %arg29: tensor<20x20xf32>, %arg30: tensor<20x20xf32>, %arg31: tensor<20x20xf32>):
      %130 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %131 = "stablehlo.add"(%arg22, %130) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      %132 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %133 = "stablehlo.broadcast_in_dim"(%132) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %134 = "stablehlo.add"(%arg20, %133) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %135 = "stablehlo.constant"() <{value = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %136 = "stablehlo.broadcast_in_dim"(%135) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %137 = "stablehlo.add"(%arg21, %136) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %138 = "stablehlo.broadcast_in_dim"(%131) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %139 = "stablehlo.multiply"(%134, %138) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %140 = "stablehlo.multiply"(%arg23, %137) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %141 = "stablehlo.multiply"(%arg25, %139) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %142 = "stablehlo.subtract"(%140, %141) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %143 = "stablehlo.multiply"(%arg24, %137) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %144 = "stablehlo.multiply"(%arg26, %139) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %145 = "stablehlo.subtract"(%143, %144) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %146 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %147 = "stablehlo.broadcast_in_dim"(%146) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %148 = "stablehlo.compare"(%145, %147) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %149 = "stablehlo.divide"(%142, %145) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %150 = "stablehlo.subtract"(%arg18, %149) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %151 = "stablehlo.divide"(%150, %149) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %152 = "stablehlo.abs"(%151) : (tensor<20x20xf32>) -> tensor<20x20xf32>
      %153 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %154 = "stablehlo.broadcast_in_dim"(%153) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %155 = "stablehlo.select"(%148, %152, %154) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %156 = "stablehlo.select"(%148, %149, %arg18) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %157 = "stablehlo.multiply"(%arg29, %137) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %158 = "stablehlo.subtract"(%157, %arg23) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %159 = "stablehlo.multiply"(%arg27, %139) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %160 = "stablehlo.subtract"(%158, %159) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %161 = "stablehlo.broadcast_in_dim"(%131) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %162 = "stablehlo.multiply"(%arg25, %161) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %163 = "stablehlo.add"(%160, %162) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %164 = "stablehlo.multiply"(%arg30, %137) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %165 = "stablehlo.subtract"(%164, %arg24) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %166 = "stablehlo.multiply"(%arg28, %139) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %167 = "stablehlo.subtract"(%165, %166) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %168 = "stablehlo.broadcast_in_dim"(%131) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %169 = "stablehlo.multiply"(%arg26, %168) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %170 = "stablehlo.add"(%167, %169) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %171 = "stablehlo.multiply"(%156, %170) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %172 = "stablehlo.subtract"(%163, %171) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %173 = "stablehlo.divide"(%172, %145) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %174 = "stablehlo.select"(%148, %173, %arg31) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %175 = "stablehlo.subtract"(%174, %arg31) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %176 = "stablehlo.abs"(%175) : (tensor<20x20xf32>) -> tensor<20x20xf32>
      %177 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %178 = "stablehlo.broadcast_in_dim"(%177) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %179 = "stablehlo.select"(%148, %176, %178) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %180 = "stablehlo.abs"(%142) : (tensor<20x20xf32>) -> tensor<20x20xf32>
      %181 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %182 = "func.call"(%181) <{callee = @integer_pow}> : (tensor<f32>) -> tensor<f32>
      %183 = "stablehlo.broadcast_in_dim"(%182) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %184 = "stablehlo.compare"(%180, %183) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %185 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %186 = "stablehlo.broadcast_in_dim"(%185) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %187 = "stablehlo.multiply"(%arg23, %186) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %188 = "stablehlo.select"(%184, %187, %arg23) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %189 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %190 = "stablehlo.broadcast_in_dim"(%189) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %191 = "stablehlo.multiply"(%142, %190) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %192 = "stablehlo.select"(%184, %191, %142) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %193 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %194 = "stablehlo.broadcast_in_dim"(%193) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %195 = "stablehlo.multiply"(%arg24, %194) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %196 = "stablehlo.select"(%184, %195, %arg24) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %197 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %198 = "stablehlo.broadcast_in_dim"(%197) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %199 = "stablehlo.multiply"(%145, %198) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %200 = "stablehlo.select"(%184, %199, %145) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %201 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %202 = "stablehlo.broadcast_in_dim"(%201) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %203 = "stablehlo.multiply"(%arg29, %202) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %204 = "stablehlo.select"(%184, %203, %arg29) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %205 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %206 = "stablehlo.broadcast_in_dim"(%205) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %207 = "stablehlo.multiply"(%arg30, %206) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %208 = "stablehlo.select"(%184, %207, %arg30) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %209 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %210 = "stablehlo.broadcast_in_dim"(%209) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %211 = "stablehlo.multiply"(%163, %210) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %212 = "stablehlo.select"(%184, %211, %163) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %213 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %214 = "stablehlo.broadcast_in_dim"(%213) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %215 = "stablehlo.multiply"(%170, %214) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %216 = "stablehlo.select"(%184, %215, %170) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %217 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %218 = "stablehlo.broadcast_in_dim"(%217) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %219 = "stablehlo.compare"(%155, %218) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %220 = "stablehlo.and"(%arg17, %219) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
      %221 = "stablehlo.select"(%arg17, %156, %arg18) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %222 = "stablehlo.select"(%arg17, %155, %arg19) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %223 = "stablehlo.select"(%arg17, %134, %arg20) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %224 = "stablehlo.select"(%arg17, %137, %arg21) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %225 = "stablehlo.select"(%arg17, %192, %arg23) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %226 = "stablehlo.select"(%arg17, %200, %arg24) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %227 = "stablehlo.select"(%arg17, %188, %arg25) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %228 = "stablehlo.select"(%arg17, %196, %arg26) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %229 = "stablehlo.select"(%arg17, %204, %arg27) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %230 = "stablehlo.select"(%arg17, %208, %arg28) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %231 = "stablehlo.select"(%arg17, %212, %arg29) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %232 = "stablehlo.select"(%arg17, %216, %arg30) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %233 = "stablehlo.select"(%arg17, %174, %arg31) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      "stablehlo.return"(%220, %221, %222, %223, %224, %131, %225, %226, %227, %228, %229, %230, %231, %232, %233) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    }) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>) -> (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>)
    %74 = "stablehlo.multiply"(%73#1, %40) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %75 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %76 = "stablehlo.broadcast_in_dim"(%75) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %77 = "stablehlo.subtract"(%76, %74) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %78 = "stablehlo.not"(%29) : (tensor<20x20xi1>) -> tensor<20x20xi1>
    %79 = "stablehlo.and"(%44, %78) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %80 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %81 = "stablehlo.broadcast_in_dim"(%80) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %82 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %83 = "stablehlo.broadcast_in_dim"(%82) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %84 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %85 = "stablehlo.broadcast_in_dim"(%84) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %86 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %87 = "stablehlo.broadcast_in_dim"(%86) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %88:7 = "stablehlo.while"(%79, %7, %81, %83, %8, %85, %87) ({
    ^bb0(%arg8: tensor<20x20xi1>, %arg9: tensor<20x20xf32>, %arg10: tensor<20x20xf32>, %arg11: tensor<20x20xf32>, %arg12: tensor<20x20xf32>, %arg13: tensor<20x20xf32>, %arg14: tensor<20x20xf32>):
      %127 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
      %128 = "stablehlo.reduce"(%arg8, %127) <{dimensions = array<i64: 0, 1>}> ({
      ^bb0(%arg15: tensor<i1>, %arg16: tensor<i1>):
        %129 = "stablehlo.or"(%arg15, %arg16) : (tensor<i1>, tensor<i1>) -> tensor<i1>
        "stablehlo.return"(%129) : (tensor<i1>) -> ()
      }) : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%128) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg1: tensor<20x20xi1>, %arg2: tensor<20x20xf32>, %arg3: tensor<20x20xf32>, %arg4: tensor<20x20xf32>, %arg5: tensor<20x20xf32>, %arg6: tensor<20x20xf32>, %arg7: tensor<20x20xf32>):
      %103 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %104 = "stablehlo.broadcast_in_dim"(%103) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %105 = "stablehlo.add"(%arg2, %104) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %106 = "stablehlo.divide"(%arg5, %105) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %107 = "stablehlo.multiply"(%arg6, %106) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %108 = "stablehlo.multiply"(%arg3, %arg5) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %109 = "stablehlo.multiply"(%105, %105) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %110 = "stablehlo.divide"(%108, %109) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %111 = "stablehlo.subtract"(%107, %110) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %112 = "stablehlo.add"(%arg7, %111) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %113 = "stablehlo.divide"(%arg5, %105) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %114 = "stablehlo.multiply"(%arg3, %113) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %115 = "stablehlo.add"(%arg4, %114) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %116 = "stablehlo.divide"(%114, %115) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %117 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %118 = "stablehlo.broadcast_in_dim"(%117) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %119 = "stablehlo.compare"(%116, %118) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %120 = "stablehlo.and"(%arg1, %119) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
      %121 = "stablehlo.select"(%arg1, %105, %arg2) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %122 = "stablehlo.select"(%arg1, %114, %arg3) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %123 = "stablehlo.select"(%arg1, %115, %arg4) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %124 = "stablehlo.select"(%arg1, %arg5, %arg5) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %125 = "stablehlo.select"(%arg1, %111, %arg6) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %126 = "stablehlo.select"(%arg1, %112, %arg7) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      "stablehlo.return"(%120, %121, %122, %123, %124, %125, %126) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    }) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>) -> (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>)
    %89 = "stablehlo.multiply"(%88#3, %40) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %90 = "stablehlo.divide"(%89, %7) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %91 = "stablehlo.select"(%29, %77, %90) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %92 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %93 = "stablehlo.broadcast_in_dim"(%92) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %94 = "stablehlo.select"(%14, %93, %91) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %95 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %96 = "stablehlo.broadcast_in_dim"(%95) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %97 = "stablehlo.select"(%17, %96, %94) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %98 = "stablehlo.or"(%24, %11) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %99 = "stablehlo.constant"() <{value = dense<0x7FC00000> : tensor<f32>}> : () -> tensor<f32>
    %100 = "stablehlo.broadcast_in_dim"(%99) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %101 = "stablehlo.select"(%98, %100, %97) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %102 = "stablehlo.convert"(%101) : (tensor<20x20xf32>) -> tensor<20x20xf16>
    "stablehlo.custom_call"(%102, %6) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<20x20xf16>, tensor<20x20xf16>) -> ()
    "func.return"(%102) : (tensor<20x20xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<20x20xf16>, tensor<20x20xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %3 = "stablehlo.constant"() <{value = dense<"0xE2C59AAF71C54CC0B0C2BDBE57C447378537CF417E3E2CC455424B42E8B891C250C1D444E33AF4BEC9C0E8BC7E3D593F0CC42FBCA0BEBC40CBC1FAB800B1113ED946263D7D3BE03F464634C34F3E1F43D6BE7B408539EBC7A642413D01403C2C51B6E4B6BB3D7743C83F0DC7C33F2540A93651C0E33E8D3CE4C32D384440AEC350BCADBF69456B3B0FC24ABBC13437BCF943FEBD07C0653D1235F4B203BF31B7E2C7883A9E3F81C5F63F1F3CA7C14DC3B2BD83C2CFC1E9B247C76840DDBDF844DD3ABF42414328BCD4B73046134326B7FDBEEFC0F7BF6C438636E7C1DA33A1C24638EE4420458544EEC08A438AC0833BFCC40C3E50BE454428C26C4475BE4A407542B03EB043EFBA5D44B13E67393CBF1FC3014243C5B3BEDDC4BA4547BF6848213A5B42D2C1E7C11543DF3EEABC70B46DC0D8318FB8AE3F483D15B48D3AAA382FB832C23042D9BF4C3D313416C664BCFDC1BEBC66276E4333BC433991C0E8C03F3CE4BD483FF199ED3CC34221C406C4404200C782AAA4B8D44479C3FD3E1639A8430AC81046B1468BBE3D40A63F4D4058C44640AC443A4407C0D643D03F623EC03DEDBBA7C010BF9BC606C201C458C65FC56CC0522D1FC40CC1FCB965BA88C70DC0383B773D964588C189C6C8BF4FC4B741E1C5AC458A3D0AC2BF413EC1D53A6DC8B1C319C05844B3C1C13E893FBAC269C3C02E2141393D2A4003C427C536C4F9C06DAD44C2583F48BCA7C03A44F13E6E31A2310ABDD5C417419BB87441B8C4933E7540B5C2893D7FBC46C647B9404446C063AA45C543C55FC2CE391340BD430136DFBB4DC69042EFBD3FB701347F3EDA3F59C03145C4BCE7BC75B9A946BE331644BF290542EC4060C196BF24C005BE8BBE0C436C406C40F143EF3F92AD57C0ECBF9BB0D8393ABE8C426F427AC1EA37ECB22A352D43C0C255BC393C2C4379C169421B42AFC36832E7BA3242EC409A438DBA00425744B63A4EB027468043D238D8B7BCBF3E3E7140664755B504C616C25E4458399A427EC356BECDC3BB415FB98A45E4446DBE473B8BBDF4BD55456C40E1C112C085C425BC94C2924110C063C75ABE3ABD03B877C02FC65DBAEF443BBDB344133D2A3B212CAD3C443DB0C1823E"> : tensor<20x20xf16>}> : () -> tensor<20x20xf16>
    %4 = "stablehlo.constant"() <{value = dense<"0x3AC43F9F30C4E7B6EC32F53E24C00CC1CA3EF5320C28003B49BE892AA0BCA63DAB40F840324023BC61B3A3C062433A476D44A7B65D41663D333A93BF98C5464054B7D744DD4097A532403AC412400FC54CC61EBDF1C5DFC1844684B874B7D4C759283141593FB3C0FCC1F543FE4487396237F1C0C54448BF944356413F44353B01C060B6E0C00FBD9EC345446EBE973C37407247C7BE44C27CB8F244D8463D3B87B42AB89439FCC5D4B8A84268C105BCA83D5BBB8BC072C0D34534BF7DC381BEEF42FCB8CE3F1AC4CC3F52C4EBC1333E3342D83F03BFC5BCDA38E5C3374350BA5431E1450B40AF3DA83DD1BDF740284238C1643C764208357B446AC55EBE5640DEB6A7BC8DC00F398FC05C427BC1E2C32341E1B445C129C5D3C6A941D73D523F324420C280C258C2EA441EB76342ECBE6C4040C426C21A3ACE3A3FBEE3BC56C1F0B829C246C3E2C032B263C272C310C41C3FC7AF40440EAEA540B14109427EBECAC39740BEBD623E063E74C410BB65C109C5D1BD9DB93B39823F48C403C57D9903412AC3F04431424B3E79C42A44EB43BE352FBEFF31F43D43C539309146633918BE2A3C8FC0B3C141C1B42B8444E13A552C94B8014244C85AC1AAC059C79B478B446C3E8AC1C43217BE5F403C4037BACE43EDAF9CBD50BEAFA073BF3C3FB8B919C1313424C5FAB37B489FB965C8A1C41934D5A8643EBFC75B34B141E1B07F439C455D4434C1FF3C63C2D03FE331713C82BBEF42DF37C2418147CEBAD93C3EBA17C0ACC6D6BC47BE09B2B74042406D3884C24B41CCA820C2C53C23C0ABC56DC03F4134C575A72BB64DC1CABB9237363D3EC3BB41DF39F739AD3B1040F3C3DBBB40B9C23E3343DB45F22E47C2983F70B1A14477C506BE79B5CC3CADBA9D3A12C59FB57337D943C7B859B287BA094700B06E45BF430DC0C23A874496C0503E11B945C1F041E6BF6B4425BE973EBB3DD8368A405FBC50314BC45D36EF290EC2EEC4B23F5842EC3F664193B886C26640CD346F3FB1C1503CAD44D8C057B87440813C4A34AFC56ABA9D3AE0C022C5D1BCE042C1B110BD31C2AABA7F39D7C2AFB7E93C12BE0DC84244923D1540E8AFBE3E54C6FABDC9C485B1F63A4B4316381D438AA7"> : tensor<20x20xf16>}> : () -> tensor<20x20xf16>
    "func.return"(%3, %4) : (tensor<20x20xf16>, tensor<20x20xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %2 = "stablehlo.constant"() <{value = dense<"0x007E007E007E007E007E007E007E007E853BC9171319007E007EC000007E007E007EF02F3C3B007E007E007E9A3BF73B007E007E007E9534007E007E007E193A007EE23B603B007E8023007ECC39007E007E007E007E007E853B007E007E007E007E007EC539007E007E007EB33B6B30C339007EBD3B007E007ED23B4B3B007E007E007E007E007E007E007E007E007E4531007E007E007E007E007E007E007E007E007EA431007E007EB23B007E007E007E007E007E007E007E007E007E007ED13B007E0132007E007E007E007E007E007E007E007E007E443A007EFB3B007E8936AD391B2A6D27007E007E007EAD3B007E6737007E0409007E007E007EC638007E007E007E007E007E0B3B007E007E007E007E007E007E007E882D007E8C09EE3B007E007E007E5D3A007E007E007E007E007E007E4D32ED36007E007E007E007E007E007E007E007E007E007E007E007E007E003C007E007EC93B007E007E007E007E007E007E9E39007E007E007E007E007E007E007E8F2A007E007E007E6634007ECF35952A007E007E6D3B1A3B007E007E5300432A007EFF00EC3B0234007E007E007E007E007E007E007E007E007E007EFD3B007E007E007E007E007E007E9F3A007E0700007E007E007E007E2F3A007E007E007E007E007E007E007E007E007E007E007E007E007E007E007E007E007E0435007E9526007E007E007E007E007E007E8336007E007E4C013636007EFC3B007E007EE93B007E5931007E007E007E007E007E007E007E007E0716007E007E007E007E007E007E007E007EE33B007E007E007E007E007EB53AFC37007E007E5A11007E007E007E007E007E007EF93B9039B43B007E007E007E007E007E007E007E007E7E28007E007E007E007E007EEB3B007E007E007E007E007E007EFB3B007E007E007E007E9A2F007E007E7538007EFF3B007EB832A0348016007E007E9500007E007E0000007E007E007E007EBD399839007E007E007E007E392D007E212C007E007E007EAF36007E1D00007E007ED738007E007E007E953A007E007E007E007E007E007E007E007E007E007E007E007E007E007EFA27007E007E007E007EDA3BB63B3C34007E007E"> : tensor<20x20xf16>}> : () -> tensor<20x20xf16>
    "func.return"(%2) : (tensor<20x20xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<f32>) -> tensor<f32>, sym_name = "integer_pow", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<f32>):
    %0 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %1 = "stablehlo.divide"(%0, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "func.return"(%1) : (tensor<f32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

