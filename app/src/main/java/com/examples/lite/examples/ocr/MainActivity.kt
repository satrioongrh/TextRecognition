/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
*/

package com.examples.lite.examples.ocr

import android.app.Activity
import android.content.ClipData
import android.content.Context
import android.content.Intent
import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.text.ClipboardManager
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import androidx.lifecycle.Observer
import androidx.lifecycle.ViewModelProvider.AndroidViewModelFactory
import com.bumptech.glide.Glide
import com.github.dhaval2404.imagepicker.ImagePicker
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.async
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import org.tensorflow.lite.examples.ocr.R
import java.util.concurrent.Executors


private const val TAG = "MainActivity"

class MainActivity : AppCompatActivity() {

    private var FOTO_URI: Uri? = null

    private lateinit var viewModel: MLExecutionViewModel
    private lateinit var resultImageView: ImageView
    private lateinit var tfImageView: ImageView
    private lateinit var runButton: Button
    private lateinit var copyButton: ImageView
    private lateinit var tvResult: TextView
    private lateinit var tvRes: TextView
    private lateinit var divRes: LinearLayout

    private var useGPU = false
    private var ocrModel: OCRModelExecutor? = null
    private val inferenceThread = Executors.newSingleThreadExecutor().asCoroutineDispatcher()
    private val mainScope = MainScope()
    private val mutex = Mutex()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.tfe_is_activity_main)

        val toolbar: Toolbar = findViewById(R.id.toolbar)
        setSupportActionBar(toolbar)
        supportActionBar?.setDisplayShowTitleEnabled(false)

        tfImageView = findViewById(R.id.tf_imageview)

        tfImageView.setOnClickListener {
            ImagePicker.with(this)
                .crop()
                .start()
        }

        resultImageView = findViewById(R.id.result_imageview)
        copyButton = findViewById(R.id.btn_copy)
        tvResult = findViewById(R.id.tv_result)
        tvRes = findViewById(R.id.tv_res)
        divRes = findViewById(R.id.div_res)

        viewModel = AndroidViewModelFactory(application).create(MLExecutionViewModel::class.java)
        viewModel.resultingBitmap.observe(
            this,
            Observer { resultImage ->
                if (resultImage != null) {
                    updateUIWithResults(resultImage)
                }
                enableControls(true)
            }
        )

        mainScope.async(inferenceThread) { createModelExecutor(useGPU) }

        runButton = findViewById(R.id.rerun_button)
        runButton.setOnClickListener {
            enableControls(false)

            mainScope.async(inferenceThread) {
                mutex.withLock {
                    if (ocrModel != null) {
                        viewModel.onApplyModel(
                            baseContext,
                            FOTO_URI,
                            ocrModel,
                            inferenceThread
                        )
                    } else {
                        Log.d(
                            TAG,
                            "Skipping running OCR since the ocrModel has not been properly initialized ..."
                        )
                    }
                }
            }
        }

        copyButton.setOnClickListener {
            setClipboard(this, tvResult.text.toString())
            if (tvResult.text.toString() != "" || tvResult.text.toString() != null){
                Toast.makeText(this, "teks disalin", Toast.LENGTH_SHORT).show()
            }
        }

//        setChipsToLogView(HashMap<String, Int>())
        enableControls(true)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == Activity.RESULT_OK && requestCode == ImagePicker.REQUEST_CODE && data != null) {
            val layoutParams = tfImageView.layoutParams as LinearLayout.LayoutParams
            layoutParams.width = LinearLayout.LayoutParams.MATCH_PARENT
            tfImageView.layoutParams = layoutParams
            FOTO_URI = data.data

            Glide.with(this)
                .load(FOTO_URI)
                .into(tfImageView)
        } else if (resultCode == ImagePicker.RESULT_ERROR) {
            Toast.makeText(this, ImagePicker.getError(data), Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(this, "membatalkan ambil gambar", Toast.LENGTH_SHORT).show()
        }
    }

    private suspend fun createModelExecutor(useGPU: Boolean) {
        mutex.withLock {
            if (ocrModel != null) {
                ocrModel!!.close()
                ocrModel = null
            }
            try {
                ocrModel = OCRModelExecutor(this, useGPU)
            } catch (e: Exception) {
                Log.e(TAG, "Fail to create OCRModelExecutor: ${e.message}")
            }
        }
    }

    private fun getColorStateListForChip(color: Int): ColorStateList {
        val states =
            arrayOf(
                intArrayOf(android.R.attr.state_enabled), // enabled
                intArrayOf(android.R.attr.state_pressed) // pressed
            )

        val colors = intArrayOf(color, color)
        return ColorStateList(states, colors)
    }

    private fun setImageView(imageView: ImageView, image: Bitmap) {
        copyButton.visibility = View.VISIBLE
        tvRes.visibility = View.VISIBLE
        divRes.visibility = View.VISIBLE
        imageView.visibility = View.VISIBLE
        Glide.with(baseContext).load(image).override(250, 250).fitCenter().into(imageView)
    }

    private fun updateUIWithResults(modelExecutionResult: ModelExecutionResult) {
        setImageView(resultImageView, modelExecutionResult.bitmapResult)

        displayTextFromHashMap(tvResult, modelExecutionResult.itemsFound)
//        setChipsToLogView(modelExecutionResult.itemsFound)
        enableControls(true)
    }

    private fun enableControls(enable: Boolean) {
        runButton.isEnabled = enable
    }

    private fun setClipboard(context: Context, text: String) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.HONEYCOMB) {
            val clipboard = context.getSystemService(CLIPBOARD_SERVICE) as ClipboardManager
            clipboard.text = text
        } else {
            val clipboard =
                context.getSystemService(CLIPBOARD_SERVICE) as android.content.ClipboardManager
            val clip = ClipData.newPlainText("Copied Text", text)
            clipboard.setPrimaryClip(clip)
        }
    }

    private fun displayTextFromHashMap(textView: TextView, hashMap: Map<String, Int>) {
        val stringBuilder = StringBuilder()

        for ((word, _) in hashMap) {
            stringBuilder.append("$word ")
        }

        textView.text = stringBuilder.toString()
    }



}
